# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
import argparse
import copy
import os
from typing import Dict, Optional, Set

try:
    import rclpy  # type: ignore[import-not-found]
    from mcap.reader import make_reader  # type: ignore[import-not-found]
    from mcap.writer import Writer  # type: ignore[import-not-found]
    from mcap_ros2.decoder import DecoderFactory  # type: ignore[import-not-found]
    from rclpy.serialization import serialize_message  # type: ignore[import-not-found]
    from rosidl_runtime_py.utilities import get_message  # type: ignore[import-not-found]
    from tqdm import tqdm  # type: ignore[import-not-found]
except ImportError as exc:
    raise ImportError(
        "Required dependencies 'mcap', 'mcap_ros2', 'rclpy', 'rosidl_runtime_py', and 'tqdm' are not available. "
        "Install them with 'pip install mcap mcap-ros2-support rclpy rosidl-runtime-py tqdm'."
    ) from exc


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter MCAP ROS2 bag: remove tf with map_center, all sem_map topics, and elevation_map topics",
    )
    parser.add_argument("--bag", required=True, type=str, help="Path to input .mcap file")
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Path to output .mcap file (default: '<bag_name>_filtered.mcap')",
    )
    parser.add_argument(
        "--frame-id",
        type=str,
        default="map_center",
        help="Frame ID to remove from tf tree (default: 'map_center')",
    )
    parser.add_argument(
        "--tf-topic",
        type=str,
        default="/tf",
        help="TF topic to filter (default: '/tf')",
    )
    parser.add_argument(
        "--tf-static-topic",
        type=str,
        default="/tf_static",
        help="TF static topic to filter (default: '/tf_static')",
    )
    parser.add_argument(
        "--sem-map-prefix",
        type=str,
        default="sem_map",
        help="Topic prefix to filter out (default: 'sem_map')",
    )
    parser.add_argument(
        "--elevation-map-prefix",
        type=str,
        default="elevation_map",
        help="Topic prefix to filter out (default: 'elevation_map')",
    )
    return parser.parse_args()


def validate_inputs(bag_path: str) -> str:
    bag_path = os.path.abspath(bag_path)
    if not os.path.isfile(bag_path):
        raise FileNotFoundError(f"Bag file not found: {bag_path}")
    if not bag_path.endswith(".mcap"):
        raise ValueError("Input bag must be an .mcap file")
    return bag_path


def should_filter_transform(transform_stamped, frame_id_to_remove: str) -> bool:
    """Check if a transform should be filtered out."""
    header_frame_id = getattr(transform_stamped.header, "frame_id", None)
    child_frame_id = getattr(transform_stamped, "child_frame_id", None)

    return (header_frame_id == frame_id_to_remove or
            child_frame_id == frame_id_to_remove)


def convert_to_ros2_message(decoded_msg, message_type_name: str):
    """Convert a decoded message to a proper ROS2 message instance."""
    msg_class = get_message(message_type_name)
    ros2_msg = msg_class()

    # For TFMessage, we just need to copy the transforms list
    if hasattr(decoded_msg, "transforms") and hasattr(ros2_msg, "transforms"):
        # Copy transforms - they should already be proper ROS2 message instances
        # or we need to convert them
        transforms = []
        for transform in decoded_msg.transforms:
            # Check if it's already a proper ROS2 message
            if hasattr(transform, '_TYPE_SUPPORT') or hasattr(type(transform), '_TYPE_SUPPORT'):
                transforms.append(transform)
            else:
                # Convert TransformStamped with proper nested message handling
                from geometry_msgs.msg import TransformStamped, Transform, Vector3, Quaternion
                from std_msgs.msg import Header
                new_transform = TransformStamped()

                # Convert header
                if hasattr(transform, 'header'):
                    header = Header()
                    header_obj = getattr(transform, 'header')
                    if hasattr(header_obj, 'stamp'):
                        from builtin_interfaces.msg import Time
                        stamp = Time()
                        stamp_obj = getattr(header_obj, 'stamp')
                        if hasattr(stamp_obj, 'sec'):
                            stamp.sec = getattr(stamp_obj, 'sec', 0)
                        if hasattr(stamp_obj, 'nanosec'):
                            stamp.nanosec = getattr(stamp_obj, 'nanosec', 0)
                        header.stamp = stamp
                    if hasattr(header_obj, 'frame_id'):
                        header.frame_id = getattr(header_obj, 'frame_id', '')
                    new_transform.header = header

                # Convert child_frame_id
                if hasattr(transform, 'child_frame_id'):
                    new_transform.child_frame_id = getattr(transform, 'child_frame_id', '')

                # Convert transform
                if hasattr(transform, 'transform'):
                    transform_obj = getattr(transform, 'transform')
                    new_transform_obj = Transform()

                    # Convert translation
                    if hasattr(transform_obj, 'translation'):
                        translation = Vector3()
                        trans_obj = getattr(transform_obj, 'translation')
                        translation.x = getattr(trans_obj, 'x', 0.0)
                        translation.y = getattr(trans_obj, 'y', 0.0)
                        translation.z = getattr(trans_obj, 'z', 0.0)
                        new_transform_obj.translation = translation

                    # Convert rotation
                    if hasattr(transform_obj, 'rotation'):
                        rotation = Quaternion()
                        rot_obj = getattr(transform_obj, 'rotation')
                        rotation.x = getattr(rot_obj, 'x', 0.0)
                        rotation.y = getattr(rot_obj, 'y', 0.0)
                        rotation.z = getattr(rot_obj, 'z', 0.0)
                        rotation.w = getattr(rot_obj, 'w', 1.0)
                        new_transform_obj.rotation = rotation

                    new_transform.transform = new_transform_obj

                transforms.append(new_transform)
        ros2_msg.transforms = transforms
    else:
        # For other message types, try to copy attributes
        for attr_name in dir(decoded_msg):
            if not attr_name.startswith('_') and hasattr(ros2_msg, attr_name):
                try:
                    setattr(ros2_msg, attr_name, getattr(decoded_msg, attr_name))
                except (AttributeError, TypeError):
                    pass

    return ros2_msg


def filter_tf_message(tf_msg, frame_id_to_remove: str):
    """Filter out transforms containing the specified frame_id."""
    if not hasattr(tf_msg, "transforms"):
        return tf_msg

    filtered_transforms = [
        transform for transform in tf_msg.transforms
        if not should_filter_transform(transform, frame_id_to_remove)
    ]

    # Create a copy of the message and update transforms
    # This preserves all other attributes of the message
    filtered_msg = copy.deepcopy(tf_msg)
    filtered_msg.transforms = filtered_transforms
    return filtered_msg


def should_filter_topic(topic: str, sem_map_prefix: str, elevation_map_prefix: str) -> bool:
    """Check if a topic should be filtered out (contains sem_map or elevation_map prefix)."""
    return sem_map_prefix in topic or elevation_map_prefix in topic


def process_bag(args: argparse.Namespace) -> None:
    # Initialize rclpy for message serialization
    try:
        rclpy.init()
    except RuntimeError:
        # Already initialized, that's fine
        pass

    try:
        bag_path = validate_inputs(args.bag)

        # Determine output path
        if args.out:
            output_path = os.path.abspath(args.out)
        else:
            base_name = os.path.splitext(bag_path)[0]
            output_path = f"{base_name}_filtered.mcap"

        if os.path.exists(output_path):
            raise FileExistsError(
                f"Output file already exists: {output_path}. "
                "Please specify a different output path with --out"
            )

        decoder_factory = DecoderFactory()
        decoders: Dict[int, object] = {}

        tf_topics = {args.tf_topic, args.tf_static_topic}
        stats = {
            "total_messages": 0,
            "tf_messages": 0,
            "filtered_tf_messages": 0,
            "removed_transforms": 0,
            "filtered_messages": 0,
            "filtered_topics": set(),
        }

        print(f"Reading MCAP file: {bag_path}")
        print(f"Filtering frame_id: {args.frame_id}")
        print(f"Filtering topics with prefix: {args.sem_map_prefix}")
        print(f"Filtering topics with prefix: {args.elevation_map_prefix}")
        print(f"Output file: {output_path}")
        print()

        print("Opening MCAP file and reading summary...")
        with open(bag_path, "rb") as input_stream:
            reader = make_reader(input_stream)

            # Get available topics
            print("Reading available topics...")
            available_topics = set()
            summary = getattr(reader, "get_summary", lambda: None)()
            if summary and getattr(summary, "channels", None):
                channels = summary.channels
                if isinstance(channels, dict):
                    available_topics = {ch.topic for ch in channels.values()}
                else:
                    available_topics = {getattr(ch, "topic", "") for ch in channels}
            elif hasattr(reader, "channels"):
                available_topics = {ch.topic for ch in reader.channels.values()}

            print(f"Found {len(available_topics)} topics")

            # Identify topics to filter (sem_map and elevation_map)
            filtered_topics = {topic for topic in available_topics if should_filter_topic(topic, args.sem_map_prefix, args.elevation_map_prefix)}
            stats["filtered_topics"] = filtered_topics
            if filtered_topics:
                print(f"Found {len(filtered_topics)} topics to filter:")
                for topic in sorted(filtered_topics):
                    print(f"  - {topic}")
            else:
                print("No topics to filter found")

            # Check if tf topics exist
            missing_topics = [topic for topic in tf_topics if topic not in available_topics]
            if missing_topics:
                print(f"Warning: TF topics not found: {missing_topics}")
                print(f"Available topics: {sorted(available_topics)[:10]}...")
            print()

            # Write output MCAP file
            with open(output_path, "wb") as output_stream:
                writer = Writer(output_stream)
                writer.start()

                try:
                    # Store channel and schema info for writing
                    # Maps: (topic, schema_id) -> (writer_schema_id, writer_channel_id)
                    channel_map: Dict[tuple, tuple] = {}

                    # Use tqdm for progress indication
                    pbar = tqdm(
                        desc="Processing messages",
                        unit="msg",
                        dynamic_ncols=True,
                    )

                    for schema, channel, message in reader.iter_messages():
                        stats["total_messages"] += 1
                        pbar.update(1)

                        # Update progress bar description with stats
                        if stats["total_messages"] % 1000 == 0:
                            pbar.set_postfix({
                                "TF": stats["tf_messages"],
                                "Removed": stats["removed_transforms"],
                                "Filtered": stats["filtered_messages"]
                            })

                        # Skip all sem_map and elevation_map topics completely
                        if should_filter_topic(channel.topic, args.sem_map_prefix, args.elevation_map_prefix):
                            stats["filtered_messages"] += 1
                            continue

                        # Register schema and channel with writer if not already registered
                        key = (channel.topic, schema.id)
                        if key not in channel_map:
                            # Register schema
                            writer_schema_id = writer.register_schema(
                                name=schema.name,
                                encoding=schema.encoding,
                                data=schema.data,
                            )

                            # Register channel
                            writer_channel_id = writer.register_channel(
                                schema_id=writer_schema_id,
                                topic=channel.topic,
                                message_encoding=channel.message_encoding,
                                metadata=channel.metadata,
                            )

                            channel_map[key] = (writer_schema_id, writer_channel_id)

                        # Get writer channel ID
                        _, writer_channel_id = channel_map[key]

                        # Handle TF messages - need to decode, filter, and re-encode
                        if channel.topic in tf_topics:
                            stats["tf_messages"] += 1

                            # Initialize decoder if needed
                            if schema.id not in decoders:
                                decoder = decoder_factory.decoder_for(
                                    channel.message_encoding, schema
                                )
                                if decoder is None:
                                    # Skip messages we can't decode
                                    continue
                                decoders[schema.id] = decoder

                            # Decode the message
                            decoded_msg = decoders[schema.id](message.data)

                            # Convert to proper ROS2 message instance
                            ros_msg = convert_to_ros2_message(decoded_msg, schema.name)

                            # Count original transforms
                            original_count = 0
                            if hasattr(ros_msg, "transforms"):
                                original_count = len(ros_msg.transforms)

                            # Filter the message
                            filtered_msg = filter_tf_message(ros_msg, args.frame_id)

                            # Count filtered transforms
                            filtered_count = 0
                            if hasattr(filtered_msg, "transforms"):
                                filtered_count = len(filtered_msg.transforms)

                            removed = original_count - filtered_count
                            stats["removed_transforms"] += removed

                            if removed > 0:
                                stats["filtered_tf_messages"] += 1

                            # Skip empty TF messages (all transforms were filtered)
                            if filtered_count == 0:
                                continue

                            # Serialize the filtered message back to bytes using rclpy
                            encoded_data = serialize_message(filtered_msg)
                        else:
                            # For non-TF messages, use the original raw data
                            encoded_data = message.data

                        writer.add_message(
                            channel_id=writer_channel_id,
                            log_time=message.log_time,
                            data=encoded_data,
                            publish_time=message.publish_time,
                            sequence=message.sequence,
                        )

                finally:
                    if pbar is not None:
                        pbar.close()
                    writer.finish()

        # Print summary
        print("\nProcessing Summary:")
        print(f"Total messages processed: {stats['total_messages']}")
        print(f"TF messages processed: {stats['tf_messages']}")
        print(f"TF messages with filtered transforms: {stats['filtered_tf_messages']}")
        print(f"Total transforms removed: {stats['removed_transforms']}")
        print(f"Filtered messages (sem_map + elevation_map): {stats['filtered_messages']}")
        print(f"Filtered topics: {len(stats['filtered_topics'])}")
        if stats['filtered_topics']:
            print("Filtered topics:")
            for topic in sorted(stats['filtered_topics']):
                print(f"  - {topic}")
        print(f"\nOutput saved to: {output_path}")
        print("Filtering completed successfully!")
    finally:
        # Shutdown rclpy
        try:
            rclpy.shutdown()
        except RuntimeError:
            pass


def main() -> None:
    args = parse_arguments()
    process_bag(args)


if __name__ == "__main__":
    main()
