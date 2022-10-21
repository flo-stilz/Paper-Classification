# Scientific Paper Classification using Visual and Textual Features

<p align="center"><img src="paper & figures/Qualitative_results.jpg" width="800px"/></p>

## Introduction
This project focuses on three main tasks:
1) Keywords/Area prediction 
2) Publication Year prediction
3) Citation/Years Count Prediction

These tasks are achieved by utilizing both visual and textual features. The visual features are the front page and figures from a given publication paper. The textual features are the title and the abstract of a given publication.


## Setup + Dataset
For the setup and dataset preparation please contact [Florian Stilz](https://github.com/flo-stilz/)
from the [Technical University of Munich](https://www.tum.de/en/).

## Architecture
The architecture used for this project is seperated in three main parts namely Textual Module in green, Visual Module in blue, and the Concatenation and Classification Module in red as follows:
<p align="center"><img src="paper & figures/network_architecture.jpg" width="1000px"/></p>


## Results
To reproduce my results I provide the following commands along with the results.

<table>
    <col>
    <col>
    <colgroup span="2"></colgroup>
    <col>
    <tr>
        <th rowspan=2>Name</th>
        <th rowspan=2>Command</th>
        <th colspan=3 scope="colgroup">Overall</th>
        <th rowspan=2>Comments</th>
    </tr>
    <tr>
        <td>F1-Micro</td>
        <td>F1-Macro</td>
        <td>Accuracy</td>
    </tr>
    <tr>
        <td>ScanRefer (Baseline)</td>
        <td><pre lang="shell">python scripts/train.py 
        --use_color --lr 1e-3 --batch_size 14</pre></td>
        <td>37.05</td>
        <td>23.93</td>
        <td>23.93</td>
        <td>xyz + color + height</td>
    </tr>
    <tr>
        <td>ScanRefer with pretrained VoteNet (optimized Baseline)</td>
        <td><pre lang="shell">python scripts/train.py 
        --use_color --use_chunking 
        --use_pretrained "pretrained_VoteNet" 
        --lr 1e-3 --batch_size 14</pre></td>
        <td>37.11</td>
        <td>25.21</td>
        <td>25.21</td>
        <td>xyz + color + height</td>
    </tr>
    <tr>
        <td>Ours (pretrained 3DETR-m + GRU + vTransformer) </td>
        <td><pre lang="shell">python scripts/train.py 
        --use_color --use_chunking 
        --detection_module 3detr 
        --match_module transformer
        --use_pretrained "pretrained_3DETR"
        --no_detection </pre></td>
        <td>37.08</td>
        <td>26.56</td>
        <td>26.56</td>
        <td>xyz + color + height</td>
    </tr>

</table>

