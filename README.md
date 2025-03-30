# molformer-regression
This project explores techniques to fine-tune a chemical language model for predicting molecular lipophilicity, a key property in drug design. It was tested different strategies, including layer freezing, influence-based data selection, and parameter-efficient methods like LoRA and BitFit. Our results show that full fine-tuning delivers the best accuracy, while data selection and lightweight adaptation methods improve efficiency. This approach offers a balanced solution for molecular property prediction, with insights for future model optimization.

## Installation

```
git clone https://github.com/umutkavakli/molformer-regression.git
cd molformer-regression
pip install -e .
```

## Running Code 
You can run the code using '<b>molreg</b>' command:
```
molreg [-h] [-m {train,test}] [-d {influence,similarity}] [-t {regression,mlm,mlm_regression}] [-b BATCH_SIZE] [-e EPOCHS] [-l LEARNING_RATE] [-s {best,last}] [-w WEIGHTS] [-p {biffit,lora,ia3}] [-r RANK] [-f {partial,full}] [-x EXTERNAL_DATA]
```

<hr>
If you run the command without argument, it will only print some information dataset:

```
molreg
```

<p>
Longest input sequence in SMILES dataset: 267 <br>
Longest tokenized input sequence in SMILES dataset: 207 <br>

Random 3 data sample out of 4200: <br>
        &emsp;&emsp;Sample 1, [INPUT STRING]: CN1C(=O)N(CC2CC2)c3nn(Cc4ccnc5ccc(Cl)cc45)c(c3C1=O)c6ncnn6C | [TARGET]: 3.8 <br>
        &emsp;&emsp;Sample 2, [INPUT STRING]: O=C(CCc1ccncc1)Nc2ccc(cc2)C(=O)c3ccccc3 | [TARGET]: 3.4 <br>
        &emsp;&emsp;Sample 3, [INPUT STRING]: CCN1CCC[C@H]1CNC(=O)c2c(O)c(Cl)cc(Cl)c2OC | [TARGET]: 1.09

Training Set Size: 3360 <br>
Validation Set Size: 420 <br>
Test Set Size: 420
</p>

<hr>

Following example trains molformer model with a regression head using LoRA technique (rank=16, alpha=32) for parameter efficient training:
```
molreg -m train -t regression -b 16 -p lora -r 16 -e 10
```

<hr>


```
options:
  -m {train,test}, --mode {train,test}
  -d {influence,similarity}, --data-selection {influence,similarity}
  -t {regression,mlm,mlm_regression}, --model-type {regression,mlm,mlm_regression}
  -b BATCH_SIZE, --batch-size BATCH_SIZE
  -e EPOCHS, --epochs EPOCHS
  -l LEARNING_RATE, --learning-rate LEARNING_RATE
  -s {best,last}, --save-weights-type {best,last}
  -w WEIGHTS, --weights WEIGHTS
  -p {biffit,lora,ia3}, --peft-type {biffit,lora,ia3}
  -r RANK, --rank RANK
  -f {partial,full}, --freeze-layers {partial,full}
  -x EXTERNAL_DATA, --external-data EXTERNAL_DATA
```

## Results

### Parameter-Efficient Fine-Tuning Methods

| Method | # Parameters | MSE (Original) | MSE (Original + External) |
|--------|-------------|---------------|---------------------------|
| **Base** | 45,556,993 | 0.40 ± 0.02 | 0.38 ± 0.01 |
| **BitFit** | 1,256,449 | 0.77 ± 0.01 | 0.74 ± 0.01 |
| **LoRA (r=4, α=8)** | 1,403,137 | 0.65 ± 0.02 | 0.63 ± 0.02 |
| **LoRA (r=16, α=32)** | 2,066,689 | 0.58 ± 0.01 | 0.57 ± 0.03 |
| **LoRA (r=64, α=128)** | 4,720,897 | 0.50 ± 0.03 | 0.48 ± 0.01 |
| **IA3** | 1,209,601 | 0.78 ± 0.02 | 0.75 ± 0.02 |

Comparison of parameter-efficient fine-tuning methods with the base model, showing the number of parameters and MSE for both Original and Original + External datasets. LoRA configurations with different rank (r) and alpha (α) values demonstrate the trade-off between parameter efficiency and performance.

### Fine-Tuning Approaches

| Method | MSE |
|--------|-----|
| **Full FT** | 0.50 ± 0.03 |
| **Head-Only FT** | 0.82 ± 0.02 |
| **Partial FT** | 0.57 ± 0.02 |

Comparison of MSE across different fine-tuning approaches. Full fine-tuning (Full FT) achieves the lowest error, followed by partial fine-tuning (Partial FT). In contrast, regression head-only fine-tuning (Head-Only FT) results in the highest MSE, indicating underfitting, as the training loss did not improve.

### Impact of Pre-Training Strategies

| Model | MSE |
|--------|-----|
| **Base** | 0.51 ± 0.03 |
| **Base + MLM** | 0.40 ± 0.02 |
| **Base + MLM + Influence** | 0.35 ± 0.02 |

Impact of different pre-training strategies on model performance measured by MSE. Fine-tuning after MLM unsupervised fine-tuning significantly reduces error compared to the base model. The best performance was obtained with external data points that have the highest influence on the model.

