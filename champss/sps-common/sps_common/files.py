from minio import Minio

import numpy as np

import io
import os

np_fromfile = np.fromfile
np_tofile = np.ndarray.tofile
np_load = np.load


class BytesAsFile:
    """
    A class that represents a file-like object backed by a bytes object.
    """

    def __init__(
        self,
        client=None,
        bucket="",
        key="",
        data=b"",
    ):
        """
        Initialize a Files object.
        Args:
            client: The client object used for interacting with the storage service.
            bucket: The name of the bucket where the file is stored.
            key: The key or path of the file within the bucket.
            data: The content of the file as bytes.
        Attributes:
            client: The client object used for interacting with the storage service.
            bucket: The name of the bucket where the file is stored.
            key: The key or path of the file within the bucket.
            data: The content of the file as bytes.
            length: The length of the file content.
            position: The current position within the file.
        """
        self.client = client
        self.bucket = bucket
        self.key = key

        self.data = data
        self.length = len(data) - 1
        self.position = 0

    def __enter__(self):
        """
        Enter method for context manager.
        Overrides common file functions.
        """
        self.override_common_file_functions()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Exit method for context manager.
        Reverts common file functions.
        Args:
            exc_type: The type of the exception raised, if any.
            exc_value: The exception instance raised, if any.
            traceback: The traceback for the exception raised, if any.
        """
        self.close()

    def seek(self, offset, whence=0):
        """
        Change the current position in the file.
        Args:
            offset (int): The offset from the reference point specified by `whence`.
            whence (int, optional): The reference point from which the offset is calculated.
                Can be value of 0 (default, beginning of the file),
                1 (current position), or 2 (end of the file).
        Raises:
            ValueError: If `whence` is not a valid value.
        """
        if whence == 0:
            self.position = 0
        elif whence == 1:
            self.position = self.position
        elif whence == 2:
            self.position = self.length
        else:
            raise ValueError("Invalid value for whence")
        self.position = self.position + offset

    def read(self, count=-1):
        """
        Read and return data from the file.
        Args:
            count (int, optional): The number of bytes to read.
                If not specified or set to -1, it will read until the end of the file.
                Defaults to -1.
        Returns:
            bytes: The data read from the file.
        """
        if count == -1:
            count = self.length
        result = self.data[self.position : self.position + count]
        self.seek(count, 1)
        return result

    # Need to implement "readlines"

    def write(self, data=b""):
        """
        Write data to the file.
        Args:
            data (bytes): The data to write to the file.
        """
        data_as_bytes = data.encode("utf-8")
        data_as_a_stream = io.BytesIO(data_as_bytes)
        self.client.put_object(
            self.bucket, self.key, data_as_a_stream, length=len(data_as_a_stream)
        )

    def close(self):
        """
        Close the file.
        Resets the internal state of the file object.
        """
        # Remove large data from memory and resets the overriden functions
        self.data = b""
        self.length = 0
        self.position = 0

        self.restore_common_file_functions()

    def override_common_file_functions(self):
        """
        Temporarily overrides np.fromfile or any other common file functions.
        Common packages like spshuff use these, so instead we override them to
        take a bytes object and use, for example, np.frombuffer.
        """
        # Probably yes since it just uses bullitin.open which we override:
        # Will yaml.load work with this class?
        # Will yaml.safe_load work with this class?
        # Will json.load work with this class?

        # Need to override since numpy uses C functions directly ignoring bullitin.open:
        # np.load?
        # np.tofile?

        np.fromfile = np_frombytes

    def restore_common_file_functions(self):
        """
        Restore the original np.fromfile function or other common file functions.
        """
        np.fromfile = np_fromfile


def np_frombytes(data, dtype, count):
    """
    Convert bytes to a NumPy array of the specified data type.
    Args:
        data (file-like object): The input data to read from.
        dtype (dtype or str): The data type of the resulting NumPy array.
        count (int): The number of elements to read.
    Returns:
        numpy.ndarray: A NumPy array containing the converted data.
    """
    # Given count value is the number of elements of the given dtype
    # We just need to translate this to bytes by getting the number
    # of bytes per element for that dtype, and multiplying it by count
    if isinstance(dtype, np.dtype):
        # "dtype" could already be np.dtype
        type_size = dtype.itemsize
    else:
        # Or it could be something like np.uint8
        type_size = np.dtype(dtype).itemsize
    number_of_bytes_to_read = count * type_size
    # Thus, only put a portion of the data into memory
    buffer = data.read(number_of_bytes_to_read)
    # Similar to np.fromfile, but we already implemented "count" with above
    # buffer slicing
    return np.frombuffer(buffer, dtype=dtype)


def open_file(filepath, mode="r"):
    """
    Opens a file specified by the given filepath on.
    Will use MinIO backend if environmental variables MINIO_AK or MINIO_SK are set (the
    user's MinIO credentials set through the MinIO console), otherwise will use the
    POSIX filesystem.
    Args:
        filepath (str): The path to the file.
        mode (str, optional): The mode in which the file should be opened. Defaults to "r".
    Returns:
        file: The opened file object.
    Raises:
        None
    """
    s3_access_key = os.getenv("MINIO_AK")
    s3_secret_key = os.getenv("MINIO_SK")

    if s3_access_key is None or s3_secret_key is None:
        # No full MinIO credentials given, will use file on POSIX filesystem
        file = open(filepath, mode)

        print("Using POSIX filesystem for file opening...")

        return file
    else:
        try:
            # Get object given bucket and path, then read it into memory, and wrap it around a
            # class so it behaves like a file
            directories = filepath.split("/")
            # First item is empty string since filepath starts with '/'
            # So we use the second item, which is the root folder, for the bucket
            # (e.g. "data", without slashes)
            s3_bucket = directories[1]
            # The rest of the path, without a leading '/' or any double "/", is the key,
            # e.g. "chime/sps/raw/....."
            s3_key = "/".join(directories[2:]).replace("//", "/")

            s3_client = Minio(
                "sps-archiver1.chime:9100",  # doesn't work with http:// in the URL
                access_key=s3_access_key,
                secret_key=s3_secret_key,
                secure=False,  # our MinIO instance isn't using TLS/SSL
            )

            s3_stream = s3_client.get_object(
                s3_bucket, s3_key
            )  # HTTP stream to read from
            s3_data = s3_stream.read()  # Putting the whole binary data into memory

            s3_file = BytesAsFile(
                s3_client,
                s3_bucket,
                s3_key,
                s3_data,
            )  # Making it behave like a file

            s3_stream.close()  # Cleanup the file connection
            s3_stream.release_conn()

            print(f"Using MinIO for file opening...")

            return s3_file
        except Exception as error:
            # MinIO credentials incorrect of file does not exist, will fall back to
            # POSIX filesystem
            print(f"Error opening file from MinIO: {error}")

            file = open(filepath, mode)

            print("Using POSIX filesystem for file opening...")

            return file
