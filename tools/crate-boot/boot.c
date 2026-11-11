/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

/*
 * crate-boot: boots a Linux guest through crate's C API (include/crate.h),
 * the way an app embeds crate, rather than through crate-run's Rust API.
 *
 * The serial console is stdin and stdout. When the guest first prints a
 * shell prompt ("root@...#"), the time since start goes to stderr. With
 * --app-region the executable memory is the app's own, mapped twice here as
 * an iOS app would, and handed over with crate_config_set_jit_region.
 *
 * --display adds a screen with a keyboard and a tablet, seen through --vnc;
 * with --frames the app's own callback takes frames too, and counts them.
 */
#include <errno.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#ifdef __APPLE__
#include <mach/mach.h>
#include <mach/mach_vm.h>
#endif

#include "crate.h"

static double started;
static int prompted;
static char tail[64];

static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t ready_cond = PTHREAD_COND_INITIALIZER;
static int ready_count;
static crate_machine *machine;

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

/* Looks for "root@" followed by "#" across output chunks. */
static void watch(const uint8_t *bytes, size_t length) {
    if (prompted) return;
    for (size_t i = 0; i < length; i++) {
        size_t n = strlen(tail);
        if (n == sizeof tail - 1) {
            memmove(tail, tail + 1, n);
            n--;
        }
        tail[n] = (char)bytes[i];
        tail[n + 1] = 0;
        char *at = strstr(tail, "root@");
        if (at && strchr(at, '#')) {
            prompted = 1;
            fprintf(stderr, "\ncrate-boot: prompt after %.2f s\n", now() - started);
            return;
        }
    }
}

static void output(void *context, uint32_t console, const uint8_t *bytes, size_t length) {
    (void)context;
    (void)console;
    fwrite(bytes, 1, length, stdout);
    fflush(stdout);
    watch(bytes, length);
}

static unsigned long frames;
static uint32_t frame_width, frame_height;

static void frame(void *context, uint32_t display, const crate_frame *f) {
    (void)context;
    frames++;
    if (f->width != frame_width || f->height != frame_height || frames <= 1) {
        fprintf(stderr, "\ncrate-boot: display %u: %ux%u (frame %lu, %u bytes a row)\n",
                display, f->width, f->height, frames, f->stride);
        frame_width = f->width;
        frame_height = f->height;
    }
}

static void input_ready(void *context, uint32_t console) {
    (void)context;
    (void)console;
    pthread_mutex_lock(&lock);
    ready_count++;
    pthread_cond_signal(&ready_cond);
    pthread_mutex_unlock(&lock);
}

static const char *state_names[] = {"created", "running", "paused", "stopped"};
static const char *reason_names[] = {"", "power off", "reset", "killed", "error"};

static crate_stop_reason stop_reason;

static void state_changed(void *context, crate_state state, crate_stop_reason reason) {
    (void)context;
    if (state == CRATE_STATE_STOPPED) stop_reason = reason;
    if (reason == CRATE_STOP_ERROR)
        fprintf(stderr, "crate-boot: %s (%s: %s)\n", state_names[state], reason_names[reason],
                crate_last_error());
    else if (reason != CRATE_STOP_NONE)
        fprintf(stderr, "crate-boot: %s (%s)\n", state_names[state], reason_names[reason]);
    else
        fprintf(stderr, "crate-boot: %s\n", state_names[state]);
}

static void log_message(void *context, crate_log_level level, const char *message) {
    (void)context;
    static const char *names[] = {"error", "warning", "info", "debug"};
    fprintf(stderr, "crate: %s: %s\n", names[level], message);
}

/* Offers stdin to the console, waiting for input_ready when it is full. */
static void *feed(void *arg) {
    (void)arg;
    uint8_t buf[256];
    for (;;) {
        ssize_t n = read(0, buf, sizeof buf);
        if (n <= 0) return NULL;
        size_t done = 0;
        while (done < (size_t)n) {
            if (crate_machine_state(machine) == CRATE_STATE_STOPPED) return NULL;
            pthread_mutex_lock(&lock);
            int seen = ready_count;
            pthread_mutex_unlock(&lock);
            size_t taken = crate_console_write(machine, 0, buf + done, (size_t)n - done);
            done += taken;
            if (done < (size_t)n) {
                /* Wait for input_ready, with a timeout in case the guest is
                   not reading at all yet. */
                struct timespec until;
                clock_gettime(CLOCK_REALTIME, &until);
                until.tv_nsec += 50 * 1000000;
                if (until.tv_nsec >= 1000000000) {
                    until.tv_sec++;
                    until.tv_nsec -= 1000000000;
                }
                pthread_mutex_lock(&lock);
                while (ready_count == seen &&
                       pthread_cond_timedwait(&ready_cond, &lock, &until) != ETIMEDOUT) {
                }
                pthread_mutex_unlock(&lock);
            }
        }
    }
}

#ifdef __APPLE__
/* Two mappings of the same pages, as an app would make for iOS. */
static int app_region(size_t size, void **exec, void **write) {
    void *rw = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANON, -1, 0);
    if (rw == MAP_FAILED) return -1;
    mach_vm_address_t rx = 0;
    vm_prot_t cur, max;
    if (mach_vm_remap(mach_task_self(), &rx, size, 0, VM_FLAGS_ANYWHERE, mach_task_self(),
                      (mach_vm_address_t)rw, FALSE, &cur, &max, VM_INHERIT_NONE) != KERN_SUCCESS)
        return -1;
    if (mprotect((void *)rx, size, PROT_READ | PROT_EXEC) != 0) return -1;
    *exec = (void *)rx;
    *write = rw;
    return 0;
}
#endif

static void check(crate_status status, const char *what) {
    if (status != CRATE_OK) {
        fprintf(stderr, "crate-boot: %s: %s (status %d)\n", what, crate_last_error(), status);
        exit(1);
    }
}

static const char usage[] =
    "usage: crate-boot --kernel PATH [--initrd PATH] [--disk PATH] [--cmdline S]\n"
    "                  [--cpus N] [--memory MIB] [--forward HOSTPORT:GUESTPORT]...\n"
    "                  [--backend auto|jit|hvf] [--jit-size MIB] [--app-region MIB]\n"
    "                  [--virtio-console] [--display 2d|virgl [--modes WxH,...]\n"
    "                  [--vnc unix:PATH|tcp:HOST:PORT] [--frames]]\n";

int main(int argc, char **argv) {
    const char *kernel = NULL, *initrd = NULL, *cmdline = "console=ttyAMA0";
    const char *disks[8];
    int ndisks = 0;
    unsigned forwards[8][2];
    int nforwards = 0;
    unsigned cpus = 1, memory = 1024, jit_size = 0, region = 0, virtio = 0, own_frames = 0;
    const char *display = NULL, *modes = NULL, *vnc = NULL;
    crate_cpu_backend backend = CRATE_CPU_AUTO;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i], *v = i + 1 < argc ? argv[i + 1] : NULL;
        if (!strcmp(a, "--virtio-console")) {
            virtio = 1;
            continue;
        }
        if (!strcmp(a, "--frames")) {
            own_frames = 1;
            continue;
        }
        if (!v) {
            fputs(usage, stderr);
            return 2;
        }
        i++;
        if (!strcmp(a, "--kernel")) kernel = v;
        else if (!strcmp(a, "--initrd")) initrd = v;
        else if (!strcmp(a, "--cmdline")) cmdline = v;
        else if (!strcmp(a, "--disk") && ndisks < 8) disks[ndisks++] = v;
        else if (!strcmp(a, "--cpus")) cpus = (unsigned)atoi(v);
        else if (!strcmp(a, "--memory")) memory = (unsigned)atoi(v);
        else if (!strcmp(a, "--jit-size")) jit_size = (unsigned)atoi(v);
        else if (!strcmp(a, "--app-region")) region = (unsigned)atoi(v);
        else if (!strcmp(a, "--forward") && nforwards < 8 &&
                 sscanf(v, "%u:%u", &forwards[nforwards][0], &forwards[nforwards][1]) == 2)
            nforwards++;
        else if (!strcmp(a, "--display")) display = v;
        else if (!strcmp(a, "--modes")) modes = v;
        else if (!strcmp(a, "--vnc")) vnc = v;
        else if (!strcmp(a, "--backend"))
            backend = !strcmp(v, "jit") ? CRATE_CPU_JIT
                      : !strcmp(v, "hvf") ? CRATE_CPU_HVF
                                          : CRATE_CPU_AUTO;
        else {
            fputs(usage, stderr);
            return 2;
        }
    }
    if (!kernel) {
        fputs(usage, stderr);
        return 2;
    }
    if (crate_api_version() != ((CRATE_API_VERSION_MAJOR << 16) | CRATE_API_VERSION_MINOR)) {
        fprintf(stderr, "crate-boot: built for crate.h %d.%d, linked %#x\n",
                CRATE_API_VERSION_MAJOR, CRATE_API_VERSION_MINOR, crate_api_version());
        return 1;
    }
    crate_set_log_callback(log_message, NULL, CRATE_LOG_INFO);
    started = now();

    crate_config *config = crate_config_new("virt");
    check(config ? CRATE_OK : CRATE_ERR_INVALID_ARGUMENT, "config");
    check(crate_config_set_cpu_backend(config, backend), "backend");
    check(crate_config_set_cpus(config, cpus), "cpus");
    check(crate_config_set_memory(config, (uint64_t)memory << 20), "memory");
    check(crate_config_set_kernel(config, kernel), "kernel");
    if (initrd) check(crate_config_set_initrd(config, initrd), "initrd");
    check(crate_config_set_cmdline(config, cmdline), "cmdline");
    for (int i = 0; i < ndisks; i++) check(crate_config_add_disk(config, disks[i], 0), "disk");
    if (jit_size) check(crate_config_set_jit_size(config, (size_t)jit_size << 20), "jit size");
#ifdef __APPLE__
    if (region) {
        void *exec, *write;
        if (app_region((size_t)region << 20, &exec, &write) != 0) {
            perror("crate-boot: mapping the app's region");
            return 1;
        }
        check(crate_config_set_jit_region(config, exec, write, (size_t)region << 20), "region");
    }
#endif
    uint32_t net;
    check(crate_config_add_network(config, &net), "network");
    for (int i = 0; i < nforwards; i++)
        check(crate_config_add_port_forward(config, net, CRATE_TCP, "127.0.0.1",
                                            (uint16_t)forwards[i][0], (uint16_t)forwards[i][1]),
              "forward");
    crate_console_callbacks callbacks = {
        .size = sizeof callbacks,
        .output = output,
        .input_ready = input_ready,
    };
    uint32_t console;
    check(crate_config_add_console(config, CRATE_CONSOLE_SERIAL, &callbacks, &console),
          "console");
    if (virtio) {
        /* A second console the guest may use as hvc0; its output joins stdout. */
        check(crate_config_add_console(config, CRATE_CONSOLE_VIRTIO, &callbacks, &console),
              "virtio console");
    }
    if (display) {
        crate_display_callbacks screen = {.size = sizeof screen, .frame = frame};
        uint32_t index;
        check(crate_config_add_display(config,
                                       !strcmp(display, "virgl") ? CRATE_GPU_VIRGL : CRATE_GPU_2D,
                                       &screen, &index),
              "display");
        if (modes) {
            crate_display_mode list[45];
            size_t count = 0;
            for (const char *m = modes; m && count < 45; m = strchr(m, ',') ? strchr(m, ',') + 1 : NULL)
                if (sscanf(m, "%ux%u", &list[count].width, &list[count].height) == 2) count++;
            check(crate_config_set_display_modes(config, index, list, count), "modes");
        }
        if (vnc) check(crate_config_add_vnc(config, index, vnc), "vnc");
        check(crate_config_add_input(config, CRATE_INPUT_KEYBOARD | CRATE_INPUT_TABLET), "input");
    }
    check(crate_config_set_state_callback(config, state_changed, NULL), "state callback");

    check(crate_machine_create(config, &machine), "create");
    crate_config_free(config);
    fprintf(stderr, "crate-boot: created in %.2f s\n", now() - started);
    check(crate_machine_start(machine), "start");
    if (display && own_frames) crate_display_set_active(machine, 0, true);

    pthread_t feeder;
    pthread_create(&feeder, NULL, feed, NULL);
    pthread_detach(feeder);

    check(crate_machine_wait(machine, -1), "wait");
    if (display) fprintf(stderr, "crate-boot: %lu frames to the app\n", frames);
    /* The state callback has reported the stop by the time wait returns. */
    crate_machine_free(machine);
    switch (stop_reason) {
    case CRATE_STOP_POWER_OFF:
    case CRATE_STOP_RESET:
        return 0;
    case CRATE_STOP_KILLED:
        return 130;
    default:
        return 1;
    }
}
