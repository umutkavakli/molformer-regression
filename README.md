# molformer-regression

```
git clone https://github.com/umutkavakli/molformer-regression.git
cd molformer-regression
pip install -e .
```

### Running Code
```
molreg [-h] [-m {train,test}] [-d {influence,similarity}] [-t {regression,mlm,mlm_regression}] [-b BATCH_SIZE] [-e EPOCHS] [-l LEARNING_RATE] [-s {best,last}] [-w WEIGHTS] [-p {biffit,lora,ia3}] [-r RANK] [-f {partial,full}] [-x EXTERNAL_DATA]
```


Following example trains molformer model with a regression head using LoRA technique (rank=16, alpha=32) for parameter efficient training:
```
molreg -m train -t regression -b 16 -p lora -r 16 -e 10
```



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