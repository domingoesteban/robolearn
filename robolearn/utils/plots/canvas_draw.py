def canvas_draw(canvas, interval):
    if canvas.figure.stale:
        canvas.draw()
    canvas.start_event_loop(interval)
    return

