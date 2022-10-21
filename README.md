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
To reproduce my results I provide the following commands along with the results for the best models for each of the three tasks.


<table>
    <col>
    <col>
    <colgroup span="2"></colgroup>
    <col>
    <tr>
        <th rowspan=2>Task</th>
        <th rowspan=2>Command</th>
        <th colspan=3 scope="colgroup">Overall</th>
        <th rowspan=2>Input Features</th>
    </tr>
    <tr>
        <td>F1-Micro</td>
        <td>F1-Macro</td>
        <td>Accuracy</td>
    </tr>
    <tr>
        <td>Keywords/Area</td>
        <td><pre lang="shell">python keyword_prediction.py 
        --gpu 'GPU-NUMBER' --input title+abstract</pre></td>
        <td>78.25%</td>
        <td>58.07%</td>
        <td>57.28%</td>
        <td>Title + Abstract</td>
    </tr>
    <tr>
        <td>Publication Year</td>
        <td><pre lang="shell">python paper_classification.py 
        --gpu 'GPU-NUMBER' --input title+abstract+image
        </pre></td>
        <td>75.15%</td>
        <td>68.89%</td>
        <td>75.15%</td>
        <td>Title + Abstract + Front Page</td>
    </tr>
    <tr>
        <td>Citation/Year Count</td>
        <td><pre lang="shell">python cite_prediction.py 
        --gpu 'GPU-NUMBER' --input title+abstract+image 
        </pre></td>
        <td>54.69%</td>
        <td>46.45%</td>
        <td>54.68%</td>
        <td>Title + Abstract + Front Page</td>
    </tr>

</table>
