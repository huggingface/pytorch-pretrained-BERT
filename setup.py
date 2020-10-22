"""
Simple check list from AllenNLP repo: https://github.com/allenai/allennlp/blob/master/setup.py

To create the package for pypi.

1. Change the version in __init__.py, setup.py as well as docs/source/conf.py. Remove the master from the links in
   the new models of the README:
   (https://huggingface.co/transformers/master/model_doc/ -> https://huggingface.co/transformers/model_doc/)
   then run `make fix-copies` to fix the index of the documentation.

2. Unpin specific versions from setup.py that use a git install.

2. Commit these changes with the message: "Release: VERSION"

3. Add a tag in git to mark the release: "git tag VERSION -m'Adds tag VERSION for pypi' "
   Push the tag to git: git push --tags origin master

4. Build both the sources and the wheel. Do not change anything in setup.py between
   creating the wheel and the source distribution (obviously).

   For the wheel, run: "python setup.py bdist_wheel" in the top level directory.
   (this will build a wheel for the python version you use to build it).

   For the sources, run: "python setup.py sdist"
   You should now have a /dist directory with both .whl and .tar.gz source versions.

5. Check that everything looks correct by uploading the package to the pypi test server:

   twine upload dist/* -r pypitest
   (pypi suggest using twine as other methods upload files via plaintext.)
   You may have to specify the repository url, use the following command then:
   twine upload dist/* -r pypitest --repository-url=https://test.pypi.org/legacy/

   Check that you can install it in a virtualenv by running:
   pip install -i https://testpypi.python.org/pypi transformers

6. Upload the final version to actual pypi:
   twine upload dist/* -r pypi

7. Copy the release notes from RELEASE.md to the tag in github once everything is looking hunky-dory.

8. Add the release version to docs/source/_static/js/custom.js and .circleci/deploy.sh

9. Update README.md to redirect to correct documentation.
"""

import shutil
import os
import re
from itertools import chain
from pathlib import Path

from setuptools import find_packages, setup


# Remove stale transformers.egg-info directory to avoid https://github.com/pypa/pip/issues/5466
stale_egg_info = Path(__file__).parent / "transformers.egg-info"
if stale_egg_info.exists():
    print(
        (
            "Warning: {} exists.\n\n"
            "If you recently updated transformers to 3.0 or later, this is expected,\n"
            "but it may prevent transformers from installing in editable mode.\n\n"
            "This directory is automatically generated by Python's packaging tools.\n"
            "I will remove it now.\n\n"
            "See https://github.com/pypa/pip/issues/5466 for details.\n"
        ).format(stale_egg_info)
    )
    shutil.rmtree(stale_egg_info)


extras = {}
# helper functions to make it easier to list dependencies not as a python list, but vertically w/ optional
# built-in comments to why a certain version of the dependency is listed
def cleanup(x):
    return re.sub(r' *#.*', '', x.strip()) # comments

def to_list(buffer):
    return list(filter(None, map(cleanup, buffer.splitlines())))

def combine_targets(names):
    return list(chain(*map(extras.get, names)))

extras["ja"] = to_list("""
    fugashi>=1.0
    ipadic>=1.0.0,<2.0
    unidic>=1.0.2
    unidic_lite>=1.0.7
""")

extras["sklearn"] = to_list("""
    scikit-learn
""")

# keras2onnx and onnxconverter-common version is specific through a commit until 1.7.0 lands on pypi
extras["tf-deps"] = to_list("""
    keras2onnx
    onnxconverter-common
    # onnxconverter-common @ git+git://github.com/microsoft/onnxconverter-common.git@f64ca15989b6dc95a1f3507ff6e4c395ba12dff5#egg=onnxconverter-common,
    # keras2onnx @ git+git://github.com/onnx/keras-onnx.git@cbdc75cb950b16db7f0a67be96a278f8d2953b48#egg=keras2onnx
""")

# keras2onnx and onnxconverter-common version is specific through a commit until 1.7.0 lands on pypi
extras["tf-main"] = to_list("""
    tensorflow>=2.0
""")

extras["tf-cpu-main"] = to_list("""
    tensorflow-cpu>=2.0
""")

extras["tf"] = combine_targets(to_list("""
    tf-deps
    tf-main
"""))

extras["tf-cpu"] = combine_targets(to_list("""
    tf-cpu-main
    tf-deps
"""))

extras["torch"] = to_list("""
    torch>=1.0
""")

extras["flax"] = to_list("""
    flax==0.2.2
    jax>=0.2.0
    jaxlib==0.1.55
""")
if os.name == "nt":  # windows
    extras["flax"] = [] # jax is not supported on windows

extras["onnxruntime"] = to_list("""
    onnxruntime>=1.4.0
    onnxruntime-tools>=1.4.2
""")

extras["serving"] = to_list("""
    fastapi
    pydantic
    starlette
    uvicorn
""")

extras["sentencepiece"] = to_list("""
    sentencepiece!=0.1.92
""")

extras["retrieval"] = to_list("""
    datasets
    faiss-cpu
""")

extras["testing-base"] = to_list("""
    parameterized
    psutil
    pytest
    pytest-xdist
    timeout-decorator
""")

extras["testing"] = combine_targets(to_list("""
    testing-base
    retrieval
"""))

extras["docs"] = to_list("""
    recommonmark
    sphinx
    sphinx-copybutton
    sphinx-markdown-tables
    sphinx-rtd-theme==0.4.3 # sphinx-rtd-theme==0.5.0 introduced big changes in the style
""")

extras["tokenizers"] = to_list("""
    tokenizers==0.9.2
""")

extras["quality"] = to_list("""
    black>=20.8b1
    flake8>=3.8.3
    isort>=5.5.4
""")

extras["dev"] = combine_targets(to_list("""
    docs
    flax
    ja
    quality
    sentencepiece
    sklearn
    testing
    tf
    torch
"""))

extras["all"] = combine_targets(to_list("""
    flax
    sentencepiece
    tf
    tokenizers
    torch
"""))

# debug 
if 0:
    from pprint import pprint
    pprint(extras)

install_requires = to_list("""
    numpy
    tokenizers==0.9.2
    dataclasses;python_version<'3.7' # dataclasses for Python versions that don't have it
    packaging   # utilities from PyPA to e.g. compare versions
    filelock    # filesystem locks e.g. to prevent parallel downloads
    requests    # for downloading models over HTTPS
    tqdm>=4.27  # progress bars in model download and training scripts
    regex!=2019.12.17     # for OpenAI GPT
    sentencepiece!=0.1.92 # for SentencePiece models
    protobuf
    sacremoses # for XLM
""")

setup(
    name="transformers",
    version="3.4.0",
    author="Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Sam Shleifer, Patrick von Platen, Sylvain Gugger, Google AI Language Team Authors, Open AI team Authors, Facebook AI Authors, Carnegie Mellon University Authors",
    author_email="thomas@huggingface.co",
    description="State-of-the-art Natural Language Processing for TensorFlow 2.0 and PyTorch",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="NLP deep learning transformer pytorch tensorflow BERT GPT GPT-2 google openai CMU",
    license="Apache",
    url="https://github.com/huggingface/transformers",
    package_dir={"": "src"},
    packages=find_packages("src"),
    install_requires = install_requires,
    extras_require=extras,
    entry_points={"console_scripts": ["transformers-cli=transformers.commands.transformers_cli:main"]},
    python_requires=">=3.6.0",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
