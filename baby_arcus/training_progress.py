"""Small, newline-based progress messages for Docker logs."""
import time


def progress_line(current, target, saved, completed, elapsed, loss, stream):
    fraction = min(1., current / target) if target else 0.
    filled = int(fraction * 24)
    rate = completed / elapsed if completed and elapsed > 0 else 0.
    eta = f'{(target-current)/rate/3600:.1f}h' if rate else 'pending'
    return (f"[{'#'*filled}{'-'*(24-filled)}] {current:,}/{target:,} "
            f"({fraction:.2%}) | saved={saved:,} | {stream} | loss={loss:.4f} | "
            f"{rate:.2f} steps/s | training-only ETA={eta} (excludes setup/evaluation)")


def emit(message):
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} UTC | {message}", flush=True)
