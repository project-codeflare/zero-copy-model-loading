#
#  Copyright (c) 2021 IBM Corp.
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import setuptools

with open("package.md", "r") as fh:
    long_description = fh.read()

# read requirements from file
with open('requirements.txt') as fh:
    requirements = fh.read().splitlines()

setuptools.setup(
    name="zerocopy",
    version="0.1.0",
    author="IBM",
    author_email="frreiss@us.ibm.com",
    description="Zero-copy model loading for PyTorch and Ray.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/frreiss/zero-copy-model-loading",
    install_requires=requirements,
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
    ],
    python_requires='>=3.7',
    package_data={"": ["LICENSE.txt"]}, 
    include_package_data=True
)
