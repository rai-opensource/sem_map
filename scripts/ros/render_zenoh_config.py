#!/usr/bin/env python3
# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
import argparse
import os

from jinja2 import Environment, FileSystemLoader


def main():
    """
    Renders the Zenoh config template with the specified router IP
    """
    parser = argparse.ArgumentParser(description="Generate Zenoh session config from template")

    parser.add_argument("--output-file", type=str, required=True, help="Where to write the rendered config file")
    parser.add_argument("--router-ip", type=str, default="127.0.0.1", help="IP address of the Zenoh router")

    args = parser.parse_args()

    # Template is in scripts/config directory
    script_dir = os.path.dirname(__file__)
    template_dir = os.path.join(script_dir, "..", "config")
    env = Environment(loader=FileSystemLoader(template_dir))

    template_file = "zenoh-session-config.json5.jinja2"
    template = env.get_template(template_file)

    context = {
        "router_ip": args.router_ip,
    }
    output = template.render(context)

    with open(args.output_file, "w") as f:
        f.write(output)


if __name__ == "__main__":
    main()
