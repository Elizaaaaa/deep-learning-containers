import datetime
import os
import logging
import random
import sys

import boto3
from botocore.config import Config
import docker
from fabric import Connection
import pytest

from test import test_utils
from test.test_utils import DEFAULT_REGION, UBUNTU_16_BASE_DLAMI, KEYS_TO_DESTROY_FILE
import test.test_utils.ec2 as ec2_utils

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
LOGGER.addHandler(logging.StreamHandler(sys.stderr))

# Immutable constant for framework specific image fixtures
FRAMEWORK_FIXTURES = (
    "pytorch_inference",
    "pytorch_training",
    "mxnet_inference",
    "mxnet_training",
    "tensorflow_inference",
    "tensorflow_training",
    "training",
    "inference",
    "gpu",
    "cpu"
)

# Ignore container_tests collection, as they will be called separately from test functions
collect_ignore = [os.path.join("container_tests", "*")]


def pytest_addoption(parser):
    default_images = test_utils.get_dlc_images()
    parser.addoption(
        "--images", default=default_images.split(" "), nargs="+", help="Specify image(s) to run",
    )
    parser.addoption(
        "--canary", action="store_true", default=False, help="Run canary tests",
    )


@pytest.fixture(scope="function")
def num_nodes(request):
    return request.param


@pytest.fixture(scope="session")
def region():
    return os.getenv("AWS_REGION", DEFAULT_REGION)


@pytest.fixture(scope="session")
def docker_client(region):
    test_utils.run_subprocess_cmd(
        f"$(aws ecr get-login --no-include-email --region {region})", failure="Failed to log into ECR.",
    )
    return docker.from_env()


@pytest.fixture(scope="session")
def dlc_images(request):
    return request.config.getoption("--images")


@pytest.fixture(scope="session")
def pull_images(docker_client, dlc_images):
    for image in dlc_images:
        docker_client.images.pull(image)


@pytest.fixture(scope="session")
def cpu_only():
    pass


@pytest.fixture(scope="session")
def gpu_only():
    pass


@pytest.fixture(scope="session")
def py3_only():
    pass


@pytest.fixture(scope="session")
def example_only():
    pass


def pytest_configure(config):
    # register canary marker
    config.addinivalue_line(
        "markers", "canary(message): mark test to run as a part of canary tests."
    )


def pytest_runtest_setup(item):
    if item.config.getoption("--canary"):
        canary_opts = [mark for mark in item.iter_markers(name="canary")]
        if not canary_opts:
            pytest.skip("Skipping non-canary tests")


def pytest_generate_tests(metafunc):
    images = metafunc.config.getoption("--images")

    # Parametrize framework specific tests
    for fixture in FRAMEWORK_FIXTURES:
        if fixture in metafunc.fixturenames:
            lookup = fixture.replace("_", "-")
            images_to_parametrize = []
            for image in images:
                if lookup in image:
                    is_example_lookup = "example_only" in metafunc.fixturenames and "example" in image
                    is_standard_lookup = "example_only" not in metafunc.fixturenames and "example" not in image
                    if is_example_lookup or is_standard_lookup:
                        if "cpu_only" in metafunc.fixturenames and "cpu" in image:
                            images_to_parametrize.append(image)
                        elif "gpu_only" in metafunc.fixturenames and "gpu" in image:
                            images_to_parametrize.append(image)
                        elif "cpu_only" not in metafunc.fixturenames and "gpu_only" not in metafunc.fixturenames:
                            images_to_parametrize.append(image)

            # Remove all images tagged as "py2" if py3_only is a fixture
            if images_to_parametrize and "py3_only" in metafunc.fixturenames:
                images_to_parametrize = [py3_image for py3_image in images_to_parametrize if "py2" not in py3_image]

    # Parametrize for framework agnostic tests, i.e. sanity
    if "image" in metafunc.fixturenames:
        metafunc.parametrize("image", images)
