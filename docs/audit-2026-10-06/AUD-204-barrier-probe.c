/* AUD-204 reachability probe for `FileJournal`'s durability barrier.
 *
 * The question the handover carried forward was whether the barrier can fail at
 * all, because `ContinuityEngine.swift:128` and `:147` discard whatever
 * `flush()` throws. This measures it against a real descriptor on a real volume
 * with the volume taken away underneath it:
 *
 *   hdiutil create -size 16m -type SPARSEBUNDLE -fs APFS -volname AUD204PROBE img
 *   hdiutil attach -nobrowse "$PWD/img.sparsebundle"          # -> /Volumes/AUD204PROBE
 *   cc -O0 AUD-204-barrier-probe.c -o /tmp/aud204/probe
 *   /tmp/aud204/probe /Volumes/AUD204PROBE/journal.probe \
 *       "hdiutil detach -force /Volumes/AUD204PROBE"
 *
 * Measured on 2026-10-08, macOS 26, the APFS image's backing store on the boot
 * volume:
 *
 *   F_FULLFSYNC                      = 51
 *   write(2)          before eject   = 17 bytes, errno 0
 *   fcntl(F_FULLFSYNC) after eject   = -1, errno 9  EBadf -- Bad file descriptor
 *   fsync()           after eject    =  0, errno 0            (reports success)
 *   write(2)          after eject    = -1, errno 5  EIO       -- Input/output error
 *   fsync()           after that     =  0, errno 0            (reports success)
 *   re-attached file                 = the 17 pre-eject bytes, and nothing more
 *
 * So the barrier's only failure signal is the `fsync` errno (Journal.swift:194-197
 * falls back from an unsupported F_FULLFSYNC to `fsync`, and takes the barrier as
 * successful when it returns 0), and `fsync` answers 0 for a descriptor whose
 * volume is gone. What does fail is the append: EIO from `write(2)`, which
 * `writeFully` throws and the session-log observer records through
 * `journalWriteFailed` -- the channel AUD-201 and AUD-202 were written about.
 * The durability promise in this scenario is therefore kept by the append path,
 * not by the barrier, and the two `try? await flush()` sites swallow an error
 * that no seam available on this machine produces. That is why this row was
 * measured and not filed.
 */
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <file-on-mounted-volume> <detach-command>\n", argv[0]);
        return 2;
    }
    printf("F_FULLFSYNC = %d\n", F_FULLFSYNC);
    int fd = open(argv[1], O_RDWR | O_CREAT | O_APPEND | O_CLOEXEC, 0600);
    if (fd < 0) {
        perror("open");
        return 2;
    }
    const char *before = "{\"kind\":\"probe\"}\n";
    errno = 0;
    ssize_t w = write(fd, before, strlen(before));
    printf("write -> %zd, errno=%d (%s)\n", w, errno, strerror(errno));
    fflush(stdout);
    if (system(argv[2]) != 0) {
        printf("detach command failed\n");
        return 2;
    }
    errno = 0;
    int r = fcntl(fd, F_FULLFSYNC);
    int e1 = errno;
    printf("fcntl(F_FULLFSYNC) -> %d, errno=%d (%s)\n", r, e1, strerror(e1));
    errno = 0;
    int s = fsync(fd);
    int e2 = errno;
    printf("fsync -> %d, errno=%d (%s)\n", s, e2, strerror(e2));
    const char *after = "{\"kind\":\"written-after-the-volume-is-gone\"}\n";
    errno = 0;
    ssize_t w2 = write(fd, after, strlen(after));
    int e3 = errno;
    printf("write-after-eject -> %zd, errno=%d (%s)\n", w2, e3, strerror(e3));
    errno = 0;
    int s2 = fsync(fd);
    printf("fsync-after-eject -> %d, errno=%d (%s)\n", s2, errno, strerror(errno));
    printf("BARRIER REPORTS FAILURE: %s\n", (r == -1 && s != 0) ? "yes" : "no");
    close(fd);
    return 0;
}
