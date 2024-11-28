# Custom data loader for DL4NLP BY HiTZ

# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Custom data loader to load and process SST data to obtain a label and a sentence. For the label, the 5-way classification task is simplified into a 2-way classification task (0 → negative, ;1 → positive. """


import csv
import os
import re

import datasets


_CITATION = """\
@inproceedings{socher-etal-2013-recursive,
    title = "Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank",
    author = "Socher, Richard and Perelygin, Alex and Wu, Jean and
      Chuang, Jason and Manning, Christopher D. and Ng, Andrew and Potts, Christopher",
    booktitle = "Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing",
    month = oct,
    year = "2013",
    address = "Seattle, Washington, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/D13-1170",
    pages = "1631--1642",
}
"""

_DESCRIPTION = """\
The Stanford Sentiment Treebank, the first corpus with fully labeled parse trees that allows for a
complete analysis of the compositional effects of sentiment in language.
"""

_HOMEPAGE = "https://nlp.stanford.edu/sentiment/"

_LICENSE = ""

_DEFAULT_URL = "https://nlp.stanford.edu/~socherr/stanfordSentimentTreebank.zip"
_PTB_URL = "https://nlp.stanford.edu/sentiment/trainDevTestTrees_PTB.zip"


class Sst(datasets.GeneratorBasedBuilder):
    """The Stanford Sentiment Treebank"""

    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="default", version=VERSION, description="Penn Treebank-formatted trees with labelled sub-sentences fomatted to sentence and labels."
        ),
    ]

    DEFAULT_CONFIG_NAME = "default"

    def _info(self):

        if self.config.name == "default":
            features = datasets.Features(
                {"label": datasets.Value("int32"),
                    "text": datasets.Value("string")
                }
            )

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        ptb_dir = dl_manager.download_and_extract(_PTB_URL)

        ptb_file_paths = {}
        for ptb_split in ["train", "dev", "test"]:
            ptb_file_paths[ptb_split] = {
                "ptb_filepath": os.path.join(ptb_dir, "trees/" + ptb_split + ".txt"),
            }

        if self.config.name == "default":
            return [
                datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs=ptb_file_paths["train"]),
                datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs=ptb_file_paths["dev"]),
                datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs=ptb_file_paths["test"]),
            ]

    def _generate_examples(
        self, ptb_filepath
    ):

        if self.config.name == "default":
            with open(ptb_filepath, encoding="utf-8") as fp:
                ptb_reader = csv.reader(fp, delimiter="\t", quoting=csv.QUOTE_NONE)
                easy_label_map={0:0, 1:0, 2:None, 3:1, 4:1}
                for id_, row in enumerate(ptb_reader):
                    
                    label = easy_label_map[int(row[0][1])]
                    # Strip out the parse information and the phrase labels---we don't need those here
                    text = re.sub(r'\s*(\(\d)|(\))\s*', '', row[0])
                    text = text[1:]
                    if label != None:
                        yield id_, {
                                "label": label,
                                "text": text}