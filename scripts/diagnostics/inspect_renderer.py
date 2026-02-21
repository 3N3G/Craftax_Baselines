#!/usr/bin/env python3
import inspect

import craftax.craftax.renderer as renderer


def main() -> None:
    print("module:", renderer.__file__)
    names = [
        n
        for n in dir(renderer)
        if any(k in n.lower() for k in ("text", "obs", "render"))
    ]
    print("names:", names)
    print("--- render_craftax_text source ---")
    print(inspect.getsource(renderer.render_craftax_text))


if __name__ == "__main__":
    main()
