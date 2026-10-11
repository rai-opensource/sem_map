# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
import argparse
import os
from dataclasses import dataclass
from typing import Dict, Generator, Optional, Tuple

import re

import cv2
import numpy as np

try:
    from mcap.reader import make_reader  # type: ignore[import-not-found]
    from mcap_ros2.decoder import DecoderFactory  # type: ignore[import-not-found]
except ImportError as exc:
    raise ImportError(
        "Required dependencies 'mcap' and 'mcap_ros2' are not available. "
        "Install them with 'pip install mcap mcap-ros2-support'."
    ) from exc

# Pattern to fix ROS2 schema references like "pkg/msg/Type" -> "pkg/Type"
_MSG_SLASH_RE = re.compile(r"(\b\w+)/msg/(\w+)")


SUPPORTED_ENCODINGS = {
    "rgb8": ("rgb", np.uint8, 3),
    "bgr8": ("bgr", np.uint8, 3),
    "rgba8": ("rgba", np.uint8, 4),
    "bgra8": ("bgra", np.uint8, 4),
    "mono8": ("mono", np.uint8, 1),
}


@dataclass
class DatasetStats:
    total_messages: int = 0
    first_timestamp_ns: Optional[int] = None
    last_timestamp_ns: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    encoding: Optional[str] = None

    def update(
        self,
        timestamp_ns: int,
        width: Optional[int],
        height: Optional[int],
        encoding: Optional[str],
    ) -> None:
        if self.first_timestamp_ns is None:
            self.first_timestamp_ns = timestamp_ns
        self.last_timestamp_ns = timestamp_ns
        if self.width is None and width is not None:
            self.width = width
        if self.height is None and height is not None:
            self.height = height
        if self.encoding is None and encoding is not None:
            self.encoding = encoding
        self.total_messages += 1

    def print_summary(self, topic: str) -> None:
        print("\nBag Insights:")
        print(f"Topic: {topic}")
        print(f"Messages: {self.total_messages}")
        if self.first_timestamp_ns is not None and self.last_timestamp_ns is not None:
            duration = (self.last_timestamp_ns - self.first_timestamp_ns) / 1e9
            print(f"Duration: {duration:.2f} seconds")
        if self.width and self.height:
            print(f"Resolution: {self.width}x{self.height}")
        if self.encoding:
            print(f"Encoding: {self.encoding}")
        print("")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract RGB frames from an MCAP ROS2 bag topic",
    )
    parser.add_argument("--bag", required=True, type=str, help="Path to .mcap file")
    parser.add_argument(
        "--topic",
        required=True,
        type=str,
        help="ROS topic containing sensor_msgs/msg/Image messages",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output directory for RGB images (default: '<bag_dir>/rgb')",
    )
    parser.add_argument(
        "--skip",
        type=int,
        default=0,
        help="Number of messages to skip between saves (0 = save all)",
    )
    parser.add_argument(
        "--rotate-180",
        action="store_true",
        help="Rotate extracted RGB frames 180 degrees (flip both horizontal and vertical) before saving",
    )
    return parser.parse_args()


def validate_inputs(bag_path: str, topic: str) -> Tuple[str, str]:
    bag_path = os.path.abspath(bag_path)
    if not os.path.isfile(bag_path):
        raise FileNotFoundError(f"Bag file not found: {bag_path}")
    if not bag_path.endswith(".mcap"):
        raise ValueError("Input bag must be an .mcap file")
    if not topic.startswith("/"):
        raise ValueError("Topic must start with '/'")
    return bag_path, topic


def iterate_image_messages(
    bag_path: str, topic: str
) -> Generator[Tuple[int, object], None, None]:
    decoder_factory = DecoderFactory()
    decoders: Dict[int, object] = {}
    with open(bag_path, "rb") as stream:
        reader = make_reader(stream)

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

        if available_topics and topic not in available_topics:
            formatted = "\n  ".join(sorted(available_topics))
            raise ValueError(f"Topic '{topic}' not found. Available topics:\n  {formatted}")

        for schema, channel, message in reader.iter_messages(topics=[topic]):
            if schema.id not in decoders:
                # Fix schemas that use "pkg/msg/Type" notation (not handled
                # by older mcap_ros2 versions).
                patched_schema = schema
                if schema.data and _MSG_SLASH_RE.search(schema.data.decode("utf-8", errors="replace")):
                    patched_data = _MSG_SLASH_RE.sub(r"\1/\2", schema.data.decode("utf-8"))
                    patched_schema = type(schema)(
                        id=schema.id,
                        name=_MSG_SLASH_RE.sub(r"\1/\2", schema.name),
                        encoding=schema.encoding,
                        data=patched_data.encode("utf-8"),
                    )
                decoder = decoder_factory.decoder_for(channel.message_encoding, patched_schema)
                if decoder is None:
                    raise ValueError(
                        f"No decoder available for encoding '{channel.message_encoding}' "
                        f"on topic '{channel.topic}'"
                    )
                decoders[schema.id] = decoder
            ros_msg = decoders[schema.id](message.data)
            yield message.log_time, ros_msg


def compressed_image_to_rgb_array(msg) -> np.ndarray:
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to decode compressed image (format: '{msg.format}')")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def ros_image_to_rgb_array(msg) -> np.ndarray:
    encoding = msg.encoding.lower()
    if encoding not in SUPPORTED_ENCODINGS:
        supported = ", ".join(sorted(SUPPORTED_ENCODINGS.keys()))
        raise ValueError(f"Unsupported encoding '{msg.encoding}'. Supported: {supported}")
    _, dtype, channels = SUPPORTED_ENCODINGS[encoding]
    buffer = np.frombuffer(msg.data, dtype=dtype)
    expected_size = msg.height * msg.step
    if buffer.size != expected_size:
        raise ValueError(
            f"Unexpected buffer size {buffer.size}, expected {expected_size} "
            f"(height={msg.height}, step={msg.step})"
        )
    if channels == 1:
        image = buffer.reshape(msg.height, msg.step).copy()
        image = image[:, : msg.width]
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    image = buffer.reshape(msg.height, msg.step // channels, channels).copy()
    image = image[:, : msg.width, :]
    color_mode = SUPPORTED_ENCODINGS[encoding][0]
    if color_mode == "rgb":
        return image
    if color_mode == "bgr":
        return image[:, :, ::-1]
    if color_mode == "rgba":
        return image[:, :, :3]
    if color_mode == "bgra":
        return image[:, :, [2, 1, 0]]
    raise ValueError(f"Unhandled color mode: {color_mode}")


def save_image(frame_idx: int, rgb_array: np.ndarray, output_dir: str) -> None:
    frame_name = f"{frame_idx:05d}"
    filename = os.path.join(output_dir, f"{frame_name}.png")
    bgr = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, bgr)


def process_bag(args: argparse.Namespace) -> None:
    bag_path, topic = validate_inputs(args.bag, args.topic)
    output_dir = args.out if args.out else os.path.join(os.path.dirname(bag_path), "rgb")
    os.makedirs(output_dir, exist_ok=True)
    skip = max(0, args.skip)

    stats = DatasetStats()
    saved_frames = 0
    processed = 0
    for timestamp_ns, ros_msg in iterate_image_messages(bag_path, topic):
        is_compressed = hasattr(ros_msg, "format") and not hasattr(ros_msg, "step")
        if not is_compressed and (
            not hasattr(ros_msg, "encoding") or not hasattr(ros_msg, "data")
        ):
            raise ValueError(
                f"Topic '{topic}' does not contain Image or CompressedImage messages "
                f"(received {type(ros_msg).__name__})"
            )

        width = getattr(ros_msg, "width", None)
        height = getattr(ros_msg, "height", None)
        encoding = getattr(ros_msg, "format", None) if is_compressed else getattr(ros_msg, "encoding", None)

        stats.update(timestamp_ns, width, height, encoding)
        if processed % (skip + 1) == 0:
            rgb_array = compressed_image_to_rgb_array(ros_msg) if is_compressed else ros_image_to_rgb_array(ros_msg)
            if args.rotate_180:
                rgb_array = np.flipud(np.fliplr(rgb_array))
            save_image(saved_frames, rgb_array, output_dir)
            saved_frames += 1
        processed += 1

    if stats.total_messages == 0:
        print("No messages found on the requested topic. Nothing was saved.")
        return

    stats.print_summary(topic)
    print(f"Processed messages: {processed}")
    print(f"Saved frames: {saved_frames}")
    print(f"Output directory: {output_dir}")
    print("RGB extraction completed successfully!")


def main() -> None:
    args = parse_arguments()
    process_bag(args)


if __name__ == "__main__":
    main()
