from time import perf_counter


class timing:
    def __enter__(self):
        self.time = perf_counter()
        return self

    def __exit__(self, *args):
        t = perf_counter() - self.time
        m = t // 60
        s = t % 60
        self.time = t
        self.readout = f"Runtime: {m}m {s:.1f}s"
        print(self.readout)
