from pathlib import Path

import boto3
from botocore import UNSIGNED
from botocore.config import Config

from angelcv.utils.logging_manager import get_logger

logger = get_logger(__name__)

CHECKPOINT_FILE_EXTENSIONS = (".ckpt", ".pt", ".pth", ".onnx", ".engine")


def get_project_root() -> Path:
    """
    Get the root directory of the project.

    Returns:
        Path to the project root directory
    """
    return Path(__file__).parents[2]


def resolve_file_path(
    file_path: str, extra_search_locations: list[str] | None = None, download_from_s3: bool = True
) -> Path:
    """
    Locate a file by searching in various locations and return its path. Optionally also search an S3 bucket and
    potentially download the file.

    Search order:
    1. Check if absolute path and exists
    2. Check in current working directory
    3. Check in project root directory
    4. Check in specified search locations (if provided)
    5. Check in default locations (config, model, dataset directories)
    6. Try to download from S3 storage (if applicable)

    Args:
        file_path: A string representing a file name or path
        extra_search_locations: Optional list of additional directories to search
        download_from_s3: Whether to download the file from S3 storage

    Returns:
        Path object to the found file

    Raises:
        FileNotFoundError: If the file is not found after searching all locations
    """
    # Convert to Path object
    path = Path(file_path)

    # 1. Check if it's an absolute path and exists
    if path.is_absolute() and path.exists():
        logger.debug(f"Found file at absolute path: {path}")
        return path

    # 2. Check as a relative path to the current working directory (normally workspace directory)
    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        logger.debug(f"Found file in as relative path to current working directory: {cwd_path}")
        return cwd_path

    # 3. Check in project root directory
    # Get the project root directory (where the main package is located)
    project_root = get_project_root()
    root_path = project_root / file_path
    if root_path.exists():
        logger.debug(f"Found file in project root: {root_path}")
        return root_path

    # 4. Check in user-specified search locations
    if extra_search_locations:
        for location in extra_search_locations:
            loc_path = Path(location) / file_path
            if loc_path.exists():
                logger.debug(f"Found file in specified location: {loc_path}")
                return loc_path

    # 5. Check in default locations within the project
    default_locations = [
        project_root / "angelcv" / "config" / "dataset",
        project_root / "angelcv" / "config" / "model",
    ]

    for location in default_locations:
        loc_path = location / file_path
        if loc_path.exists():
            logger.debug(f"Found file in default location: {loc_path}")
            return loc_path

    # 6. Try to download from S3 storage (for models and datasets)
    # This is a placeholder - implementation would depend on your S3 setup
    if path.name.endswith(CHECKPOINT_FILE_EXTENSIONS) and download_from_s3:
        # This should be replaced with your actual S3 download logic
        s3_path = _check_and_download_from_s3(file_path)
        if s3_path and s3_path.exists():
            logger.info(f"Downloaded file from S3: {s3_path}")
            return s3_path

    # File not found after all attempts
    logger.warning(f"Could not find file: {file_path}")
    raise FileNotFoundError(f"File not found: {file_path}")


def _check_and_download_from_s3(file_name: str) -> Path | None:
    """
    Check if a file is available in S3 storage and download it if it exists.

    Args:
        file_name: The name of the file to look for in S3

    Returns:
        Path to the downloaded file or None if not found or download failed
    """
    # S3 bucket name for the project
    s3_bucket = "angelcv"

    local_dir = get_project_root()
    local_path = local_dir / file_name

    # Check if already downloaded
    if local_path.exists():
        return local_path

    # Try to check if the file exists in S3 and download it
    try:
        # Initialize boto3 S3 client
        s3_client = boto3.client("s3", config=Config(signature_version=UNSIGNED))

        # Use a paginator to check if the file exists
        paginator = s3_client.get_paginator("list_objects_v2")
        file_exists = False

        # Iterate through the pages to find the file
        for page in paginator.paginate(Bucket=s3_bucket, Prefix=""):
            if "Contents" in page:
                for obj in page["Contents"]:
                    print(obj["Key"])
                    if obj["Key"] == file_name:
                        file_exists = True
                        break
                if file_exists:
                    break

        if file_exists:
            logger.info(f"File {file_name} found in S3 bucket, attempting download...")

            # Download the file
            s3_client.download_file(s3_bucket, file_name, str(local_path))

            if local_path.exists():
                logger.info(f"Successfully downloaded {file_name} to {local_path}")
                return local_path
            else:
                logger.warning(f"Download seemed to succeed but file not found at {local_path}")
                return None
        else:
            logger.warning(f"File {file_name} not found in S3 bucket")
            return None

    except Exception as e:
        logger.warning(f"Failed to download {file_name} from S3: {e}")
        return None


def test_resolve_file_path(input_str: str):
    try:
        result = resolve_file_path(input_str)
        print(f"Searching for {input_str}, result: {result}")
    except FileNotFoundError as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    test_resolve_file_path("yolov5s.ckpt")
    test_resolve_file_path("yolov10s.ckpt")
    test_resolve_file_path("yolov10s.yaml")
    test_resolve_file_path("dataset.yaml")
    test_resolve_file_path("coco.yaml")
