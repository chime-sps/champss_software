from sps_common.files import open_file
import builtins

builtins.open = open_file
