# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
import argparse
import os
from typing import Dict

try:
    import rclpy  # type: ignore[import-not-found]
    from builtin_interfaces.msg import Time  # type: ignore[import-not-found]
    from geometry_msgs.msg import (  # type: ignore[import-not-found]
        Quaternion,
        Transform,
        TransformStamped,
        Vector3,
    )
    from mcap.reader import make_reader  # type: ignore[import-not-found]
    from mcap.writer import Writer  # type: ignore[import-not-found]
    from mcap_ros2.decoder import DecoderFactory  # type: ignore[import-not-found]
    from rclpy.serialization import serialize_message  # type: ignore[import-not-found]
    from rosidl_runtime_py.utilities import get_message  # type: ignore[import-not-found]
    from std_msgs.msg import Header  # type: ignore[import-not-found]
    from tf2_msgs.msg import TFMessage  # type: ignore[import-not-found]
    from tqdm import tqdm  # type: ignore[import-not-found]
except ImportError as exc:
    raise ImportError(
        "Required dependencies 'mcap', 'mcap_ros2', 'rclpy', 'rosidl_runtime_py', and 'tqdm' are not available. "
        "Install them with 'pip install mcap mcap-ros2-support rclpy rosidl-runtime-py tqdm'."
    ) from exc


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add a static transform to an MCAP ROS2 bag file",
    )
    parser.add_argument("--bag", required=True, type=str, help="Path to input .mcap file")
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Path to output .mcap file (default: '<bag_name>_with_tf.mcap')",
    )
    parser.add_argument("--frame-id", required=True, type=str, help="Parent frame ID")
    parser.add_argument("--child-frame-id", required=True, type=str, help="Child frame ID")
    parser.add_argument("--x", type=float, default=0.0, help="Translation X")
    parser.add_argument("--y", type=float, default=0.0, help="Translation Y")
    parser.add_argument("--z", type=float, default=0.0, help="Translation Z")
    parser.add_argument("--qx", type=float, default=0.0, help="Rotation quaternion X")
    parser.add_argument("--qy", type=float, default=0.0, help="Rotation quaternion Y")
    parser.add_argument("--qz", type=float, default=0.0, help="Rotation quaternion Z")
    parser.add_argument("--qw", type=float, default=1.0, help="Rotation quaternion W")
    parser.add_argument(
        "--tf-static-topic",
        type=str,
        default="/tf_static",
        help="TF static topic name (default: '/tf_static')",
    )
    return parser.parse_args()


def validate_inputs(bag_path: str) -> str:
    bag_path = os.path.abspath(bag_path)
    if not os.path.isfile(bag_path):
        raise FileNotFoundError(f"Bag file not found: {bag_path}")
    if not bag_path.endswith(".mcap"):
        raise ValueError("Input bag must be an .mcap file")
    return bag_path


def build_static_transform(args: argparse.Namespace, stamp_sec: int = 0, stamp_nanosec: int = 0) -> TFMessage:
    """Build a TFMessage containing the static transform."""
    t = TransformStamped()
    t.header = Header()
    t.header.stamp = Time(sec=stamp_sec, nanosec=stamp_nanosec)
    t.header.frame_id = args.frame_id
    t.child_frame_id = args.child_frame_id
    t.transform = Transform()
    t.transform.translation = Vector3(x=args.x, y=args.y, z=args.z)
    t.transform.rotation = Quaternion(x=args.qx, y=args.qy, z=args.qz, w=args.qw)

    return TFMessage(transforms=[t])


def process_bag(args: argparse.Namespace) -> None:
    try:
        rclpy.init()
    except RuntimeError:
        pass

    try:
        bag_path = validate_inputs(args.bag)

        if args.out:
            output_path = os.path.abspath(args.out)
        else:
            base_name = os.path.splitext(bag_path)[0]
            output_path = f"{base_name}_with_tf.mcap"

        if os.path.exists(output_path):
            raise FileExistsError(
                f"Output file already exists: {output_path}. "
                "Please specify a different output path with --out"
            )

        print(f"Reading MCAP file: {bag_path}")
        print(f"Adding static transform: {args.frame_id} -> {args.child_frame_id}")
        print(f"  Translation: [{args.x}, {args.y}, {args.z}]")
        print(f"  Rotation (xyzw): [{args.qx}, {args.qy}, {args.qz}, {args.qw}]")
        print(f"Output file: {output_path}")
        print()

        with open(bag_path, "rb") as input_stream:
            reader = make_reader(input_stream)

            # Check if /tf_static topic already exists
            has_tf_static = False
            first_timestamp = None
            summary = getattr(reader, "get_summary", lambda: None)()
            if summary and getattr(summary, "channels", None):
                channels = summary.channels
                if isinstance(channels, dict):
                    has_tf_static = any(
                        ch.topic == args.tf_static_topic for ch in channels.values()
                    )
                else:
                    has_tf_static = any(
                        getattr(ch, "topic", "") == args.tf_static_topic for ch in channels
                    )

            print(f"Existing {args.tf_static_topic} topic: {'Yes' if has_tf_static else 'No'}")

            with open(output_path, "wb") as output_stream:
                writer = Writer(output_stream)
                writer.start()

                try:
                    channel_map: Dict[tuple, tuple] = {}
                    tf_static_injected = False
                    tf_static_writer_channel_id = None

                    pbar = tqdm(
                        desc="Processing messages",
                        unit="msg",
                        dynamic_ncols=True,
                    )

                    total_messages = 0

                    for schema, channel, message in reader.iter_messages():
                        total_messages += 1
                        pbar.update(1)

                        if first_timestamp is None:
                            first_timestamp = message.log_time

                        # Register schema and channel with writer if not already registered
                        key = (channel.topic, schema.id)
                        if key not in channel_map:
                            writer_schema_id = writer.register_schema(
                                name=schema.name,
                                encoding=schema.encoding,
                                data=schema.data,
                            )
                            writer_channel_id = writer.register_channel(
                                schema_id=writer_schema_id,
                                topic=channel.topic,
                                message_encoding=channel.message_encoding,
                                metadata=channel.metadata,
                            )
                            channel_map[key] = (writer_schema_id, writer_channel_id)

                        _, writer_channel_id = channel_map[key]

                        # Track the tf_static channel for injection
                        if channel.topic == args.tf_static_topic:
                            tf_static_writer_channel_id = writer_channel_id

                        # Write original message as-is
                        writer.add_message(
                            channel_id=writer_channel_id,
                            log_time=message.log_time,
                            data=message.data,
                            publish_time=message.publish_time,
                            sequence=message.sequence,
                        )

                    pbar.close()

                    # If there was no /tf_static topic, register one
                    if tf_static_writer_channel_id is None:
                        # Register the TFMessage schema using CDR encoding
                        tf_schema = get_message("tf2_msgs/msg/TFMessage")
                        # Use the ROS2 message definition schema
                        from mcap_ros2._dynamic import generate_dynamic

                        # Register schema for tf2_msgs/msg/TFMessage
                        # We need to get the schema data in ros2msg format
                        import hashlib

                        from rosidl_adapter.parser import parse_message_string

                        schema_text = (
                            "geometry_msgs/msg/TransformStamped[] transforms\n"
                        )
                        writer_schema_id = writer.register_schema(
                            name="tf2_msgs/msg/TFMessage",
                            encoding="ros2msg",
                            data=schema_text.encode(),
                        )
                        tf_static_writer_channel_id = writer.register_channel(
                            schema_id=writer_schema_id,
                            topic=args.tf_static_topic,
                            message_encoding="cdr",
                            metadata={},
                        )

                    # Inject the static transform
                    stamp_ns = first_timestamp if first_timestamp else 0
                    stamp_sec = stamp_ns // 1_000_000_000
                    stamp_nanosec = stamp_ns % 1_000_000_000

                    tf_msg = build_static_transform(args, stamp_sec=stamp_sec, stamp_nanosec=stamp_nanosec)
                    encoded_data = serialize_message(tf_msg)

                    writer.add_message(
                        channel_id=tf_static_writer_channel_id,
                        log_time=first_timestamp if first_timestamp else 0,
                        data=encoded_data,
                        publish_time=first_timestamp if first_timestamp else 0,
                        sequence=0,
                    )
                    tf_static_injected = True

                finally:
                    writer.finish()

        print(f"\nTotal messages copied: {total_messages}")
        print(f"Static transform injected: {tf_static_injected}")
        print(f"\nOutput saved to: {output_path}")
        print("Done!")

    finally:
        try:
            rclpy.shutdown()
        except RuntimeError:
            pass


def main() -> None:
    args = parse_arguments()
    process_bag(args)


if __name__ == "__main__":
    main()
