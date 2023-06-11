<a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-nd/4.0/88x31.png" /></a><br />The dataset is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/">Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License</a>.

# doc-hack

A Telegram bot for identifying diseases and doctors by a description of the symptoms.

![](media/screenshot.png)

## Evaluation results

BERT:
```javascript
LRAP = 0.5867235614792822
eval_loss = 0.1558876243572702
f1 = 0.43116946664491923
```

XLM-RoBERTa:
```javascript
LRAP = 0.610907659358978
eval_loss = 0.1509880079086035
f1 = 0.47227688389859934
```

<details>
  <summary>TensorBoard charts</summary>
  
  Red -- BERT, blue -- XLM-RoBERTa.
  
  _eval_eval_loss_
  
  ![](media/eval_eval_loss.svg)
  
  _eval_f1_
  
  ![](media/eval_f1.svg)
  
  _eval_LRAP_
  
  ![](media/eval_LRAP.svg)
  
  _loss_
  
  ![](media/loss.svg)
  
  _lr_
  
  ![](media/lr.svg)
</details>
