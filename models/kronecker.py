import subprocess


# TODO: move this to a config file
snap_dir = "./snap/snap/examples/kronem/"
command_name = "kronem"
command = snap_dir + command_name


def train():
    subprocess.call(
        [
            command,
            "-i:./data/as20graph.txt",
            "-n0:2",
            '-m:"0.9 0.6; 0.6 0.1"',
            "-ei:50",
        ],
    )


def kronecker():
    pass


if __name__ == "__main__":
    train()
