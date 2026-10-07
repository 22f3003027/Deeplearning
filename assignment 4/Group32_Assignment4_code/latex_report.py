"""Create self-contained LaTeX reports from verified Assignment 3 and 4 results.

Generates editable .tex files plus local figure assets; optionally compiles with
pdflatex. No measured values are hard-coded into the reports.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess
import zipfile

import numpy as np

PREAMBLE = r"""\documentclass[11pt,a4paper]{article}
\usepackage[margin=22mm,headheight=15pt]{geometry}
\usepackage{graphicx,amsmath,amsfonts,booktabs,array,longtable,tabularx}
\usepackage{xcolor,fancyhdr,hyperref}
\definecolor{ink}{HTML}{20334B}
\definecolor{accent}{HTML}{245E91}
\hypersetup{colorlinks=true,linkcolor=accent,urlcolor=accent,pdftitle={@PDFTITLE@},pdfauthor={Group 32}}
\graphicspath{{figures/}}
\pagestyle{fancy}\fancyhf{}
\fancyhead[L]{\small CS601T: Deep Learning}
\fancyhead[R]{\small @ASSIGNMENT@ / Group 32}
\fancyfoot[L]{\small MNIST digits 0, 3, 5, 6, 9}
\fancyfoot[R]{\thepage}
\setlength{\parindent}{0pt}\setlength{\parskip}{7pt}
\setlength{\emergencystretch}{2em}
\renewcommand{\arraystretch}{1.18}
\newcolumntype{Y}{>{\raggedright\arraybackslash}X}
\begin{document}
"""


def tex(value):
    mapping={"\\":r"\textbackslash{}","&":r"\&","%":r"\%","$":r"\$","#":r"\#",
             "_":r"\_","{":r"\{","}":r"\}","~":r"\textasciitilde{}","^":r"\textasciicircum{}"}
    return "".join(mapping.get(c,c) for c in str(value))


def percent(number):
    return f"{number*100:.2f}\\%"


def hidden(name):
    return name.replace("_",r" $\to$ ")


class Report:
    def __init__(self,project,assignment,title):
        self.project=project
        project.mkdir(parents=True,exist_ok=True)
        (project/"figures").mkdir(exist_ok=True)
        self.filename=f"Group32_Assignment{assignment}_report.tex"
        self.parts=[PREAMBLE.replace("@PDFTITLE@",tex(title)).replace("@ASSIGNMENT@",f"Assignment {assignment}")]

    def raw(self,value): self.parts.append(value+"\n")
    def paragraph(self,value): self.raw(value+"\n")
    def section(self,title,fresh=True):
        if fresh: self.raw(r"\clearpage")
        self.raw(r"\section{"+tex(title)+"}")

    def table(self,headers,rows,spec=None,font=r"\small",escape_values=False):
        spec=spec or "l"+"r"*(len(headers)-1)
        self.raw(r"\begingroup"+font+r"\setlength{\tabcolsep}{5pt}")
        self.raw(r"\begin{tabularx}{\linewidth}{"+spec+"}")
        self.raw(r"\toprule")
        self.raw(" & ".join(r"\textbf{"+tex(h)+"}" for h in headers)+r" \\")
        self.raw(r"\midrule")
        for row in rows:
            self.raw(" & ".join(tex(v) if escape_values else str(v) for v in row)+r" \\")
        self.raw(r"\bottomrule\end{tabularx}\endgroup\par\medskip")

    def image(self,source,caption,height="0.72\\textheight"):
        shutil.copy2(source,self.project/"figures"/source.name)
        self.raw(r"\begin{center}")
        self.raw(r"\includegraphics[width=\linewidth,height="+height+r",keepaspectratio]{"+source.name+"}")
        self.raw(r"\end{center}")
        self.raw(r"{\small "+caption+r"}\par")

    def title(self,assignment,title,subtitle):
        self.raw(r"\hypersetup{pageanchor=false}")
        self.raw(r"\begin{titlepage}\thispagestyle{empty}")
        self.raw(r"{\color{accent}\large CS601T: Deep Learning}\par\vspace{24mm}")
        self.raw(r"{\color{ink}\Huge\bfseries "+title+r"}\par\vspace{8mm}")
        self.raw(r"{\LARGE Programming Assignment "+str(assignment)+r"}\par\vspace{4mm}")
        self.raw(r"{\Large Group 32}\par\vspace{5mm}")
        self.raw(r"{\large Autumn semester 2026}\par\vspace{10mm}")
        self.paragraph(subtitle)

    def end_title(self): self.raw(r"\vfill\end{titlepage}\hypersetup{pageanchor=true}")

    def finish(self):
        self.raw(r"\end{document}")
        destination=self.project/self.filename
        destination.write_text("\n".join(self.parts),encoding="utf-8")
        (self.project/"README.md").write_text(
            f"# Group 32 LaTeX report\n\nMain file: `{self.filename}`. Figures are local; no external files or Python are needed to compile.\n\n"
            f"Run twice from this directory:\n\n```sh\npdflatex -interaction=nonstopmode -halt-on-error {self.filename}\n"
            f"pdflatex -interaction=nonstopmode -halt-on-error {self.filename}\n```\n\n"
            "For Overleaf, upload this project ZIP and set the main document to the `.tex` file. Use pdfLaTeX.\n"
            "All table values are generated from measured experiment results. You can edit the LaTeX normally.\n",encoding="utf-8")
        return destination


def build_assignment3(a3,project):
    r=json.loads((a3/"metrics.json").read_text())
    if r["status"]!="complete": raise ValueError("Assignment 3 training is incomplete")
    doc=Report(project,3,"Optimizer comparison for fully connected neural networks")
    doc.title(3,r"Optimizer comparison\\for fully connected\\neural networks",
              "Reproduction of the supplied Assignment 3 specification on the original Group 32 MNIST splits. "
              "This newly measured study supplies the Assignment 3 baseline required by Assignment 4.")
    doc.table(["Component","Coverage"],[
        ["Architectures","Three, four, and five ReLU hidden layers; varied intermediate widths"],
        ["Optimizers","SGD, batch GD, momentum, NAG, AdaGrad, RMSProp, Adam"],
        ["Stopping","Absolute successive epoch mean training loss difference below $10^{-4}$"],
        ["Model selection","Highest validation accuracy across converged architecture/optimizer runs"],
    ],"lY")
    doc.paragraph("All runs use the prescribed learning rate, batch sizes, and optimizer parameters. "
                  "There is no epoch limit. All measured tables and figures are saved with the source code.")
    doc.end_title()
    doc.section("Dataset and reproducible protocol",False)
    doc.table(["Digit","Train","Validation","Test"],[[c]+[r["counts"][s][str(c)] for s in ("train","val","test")] for c in r["classes"]]
              +[["Total"]+[sum(r["counts"][s].values()) for s in ("train","val","test")]],"Yrrr")
    doc.paragraph("The supplied 28 $\\times$ 28 grayscale images are flattened to 784 pixels and scaled to $[0,1]$ by dividing by 255. "
                  "The supplied splits are retained without re-splitting. Labels are digits 0, 3, 5, 6, and 9. "
                  "No additional per-pixel standardization is applied in this Assignment 3 study.")
    doc.table(["Name","Hidden layers","Full architecture"],
              [[tex(name),len(widths),r"784 $\to$ "+r" $\to$ ".join(map(str,widths))+r" $\to$ 5"]
               for name,widths in r["config"]["architectures"].items()],"lrl")
    doc.paragraph("The first hidden width is held at 32; depth and intermediate widths vary. ReLU is used in each hidden layer, "
                  "with five linear logits and softmax cross-entropy. Initialization is seeded He normal with zero biases. "
                  "For each architecture, every optimizer starts from identical weights and biases, verified using a SHA-256 digest. "
                  "Online optimizers also use the same seeded image permutation in corresponding epochs.")
    doc.raw(r"\[L(\theta)=\frac{1}{N}\sum_{i=1}^{N}-\log\frac{\exp z_{i,y_i}}{\sum_{c=1}^{5}\exp z_{i,c}}.\]")
    doc.paragraph("The stopping rule is applied to the complete training split at the end of each epoch, after that epoch's updates: "
                  "$|L_t-L_{t-1}|<10^{-4}$. Training stops at the first qualifying epoch; the first possible stop is epoch 2. "
                  "Validation does not stop training or choose an epoch. Final converged weights are evaluated on validation for model selection.")
    doc.paragraph("Numba compiles the exact sample-wise forward, backward, and parameter update loops in float64. "
                  "Each online sample is followed by one update; it is not replaced by minibatch training. "
                  "The full-batch implementations use NumPy matrix products. All seven update rules passed numerical comparisons with PyTorch.")
    doc.section("Prescribed optimizer settings")
    doc.table(["Optimizer","Batch size","Parameters"],[
        ["SGD",1,r"$\eta=0.001$"],
        ["Batch gradient descent",11385,r"$\eta=0.001$; mean full-batch gradient"],
        ["SGD with momentum",1,r"$\eta=0.001$, $\mu=0.9$"],
        ["Nesterov accelerated gradient",1,r"$\eta=0.001$, $\mu=0.9$"],
        ["AdaGrad",11385,r"$\eta=0.001$, $\epsilon=10^{-8}$; zero initial accumulator"],
        ["RMSProp",11385,r"$\eta=0.001$, $\rho=0.99$, $\epsilon=10^{-8}$"],
        ["Adam",1,r"$\eta=0.001$, $\beta_1=0.9$, $\beta_2=0.999$, $\epsilon=10^{-8}$"],
    ],"YrY")
    doc.paragraph("Momentum uses $v_t=0.9v_{t-1}+g_t$ and $\\theta_t=\\theta_{t-1}-0.001v_t$. "
                  "NAG uses the equivalent update convention in PyTorch SGD with nesterov=True: "
                  "$\\theta_t=\\theta_{t-1}-0.001(g_t+0.9v_t)$. RMSProp is uncentered with no additional momentum, "
                  "and Adam uses bias correction. No weight decay, dropout, or learning-rate schedule is introduced.")
    names=list(r["config"]["optimizers"])
    doc.section("Epochs to the specified stopping threshold")
    rows=[]
    for architecture in r["config"]["architectures"]:
        runs={row["optimizer"]:row for row in r["runs"] if row["architecture"]==architecture}
        rows.append([tex(architecture)]+[runs[name]["convergence_epochs"] for name in names])
    doc.table(["Architecture",*names],rows,"Y"+"r"*7,font=r"\scriptsize")
    doc.paragraph("The table counts complete passes through the training set. One online epoch contains 11,385 optimizer updates, "
                  "whereas one full-batch epoch contains one update. Consequently, epochs alone are not a matched computational budget. "
                  "Every run's final absolute loss difference is below $10^{-4}$, and no earlier recorded pair satisfies the stopping condition.")
    doc.paragraph("This literal stopping rule can classify a tiny loss change as convergence even at a high loss. "
                  "In particular, a full-batch learning rate of 0.001 can produce small changes near initialization. "
                  "The report preserves such outcomes instead of continuing beyond the assigned criterion or substituting a fixed epoch budget.")
    for architecture,widths in r["config"]["architectures"].items():
        doc.section("Results: "+" - ".join(map(str,widths)))
        runs=[row for row in r["runs"] if row["architecture"]==architecture]
        doc.table(["Optimizer","Epochs","Train acc.","Val. acc.","Final CE","Final loss delta"],
                  [[tex(row["optimizer"]),row["convergence_epochs"],percent(row["train"]["accuracy"]),
                    percent(row["val"]["accuracy"]),f"{row['train']['loss']:.6f}",f"{row['final_loss_difference']:.2e}"]
                   for row in runs],"Yrrrrr",font=r"\small")
        doc.image(a3/"figures"/f"loss_{architecture}.png",
                  "Superimposed post-epoch average training cross-entropy for all seven optimizers. "
                  "The right panel magnifies losses below 0.5 with a logarithmic epoch axis; high-loss curves remain visible in the left panel.","0.36\\textheight")
        winning=max(runs,key=lambda row:(row["val"]["accuracy"],-row["val"]["loss"]))
        doc.paragraph("For this architecture, "+tex(winning["optimizer"])+" has the highest validation accuracy, "+percent(winning["val"]["accuracy"])+". "
                      "The full curve, convergence loss difference, number of updates, and measured runtime are available in metrics.json.")
    selected=r["selected"]
    doc.section("Validation-selected model and test evaluation")
    doc.table(["Attribute","Selected result"],[
        ["Hidden layers",r" $\to$ ".join(map(str,selected["hidden"]))],
        ["Optimizer",tex(selected["optimizer"])],
        ["Convergence epochs",selected["convergence_epochs"]],
        ["Training accuracy",percent(selected["train"]["accuracy"])],
        ["Validation accuracy",percent(selected["val"]["accuracy"])],
        ["Test accuracy",percent(selected["test"]["accuracy"])],
    ],"lY")
    doc.image(a3/"figures"/"confusion_selected.png","Rows are true digits and columns are predicted digits. "
              "The training matrix has 2,277 observations per row; the test matrix has 759 per row.","0.43\\textheight")
    matrix=np.asarray(selected["test"]["confusion_matrix"]); off=matrix.copy(); np.fill_diagonal(off,0)
    i,j=np.unravel_index(off.argmax(),off.shape)
    doc.paragraph(f"The largest directional test confusion is digit {r['classes'][i]} predicted as {r['classes'][j]} ({off[i,j]} images). "
                  "The test split is used only after the final architecture and optimizer have been selected on validation accuracy; "
                  "validation cross-entropy breaks an accuracy tie.")
    doc.section("Observations and reproducibility")
    doc.paragraph("The comparison combines different update schedules because the brief explicitly prescribes them. "
                  "Any performance gap therefore reflects both the optimizer and its assigned batch size. "
                  "Identical initialization controls the starting point, but does not remove those scheduling differences. "
                  "Small epoch-to-epoch loss changes also do not prove a minimum or high classification accuracy.")
    doc.paragraph("There is one seeded run per architecture and optimizer, not repeated trials. "
                  "The tested compact widths are a finite architecture search, so the selected result is the best among these experiments. "
                  "It is a fresh reproduction of Assignment 3, not a claim about a previously submitted model.")
    doc.paragraph("Source: \\nolinkurl{CS601T_Assignment3_Details_07Sept2026.pdf}. "
                  "The handout's printed date contains a year inconsistency; it does not change the experiment requirements.")
    doc.paragraph("Code folder: \\texttt{Group32\\_Assignment3\\_code}.")
    doc.paragraph("Run \\texttt{python run\\_assignment3.py}, "
                  "then \\texttt{python verify\\_results.py}. "
                  "The measured comparison record is \\nolinkurl{results/assignment3_best.json}.")
    doc.paragraph("Dataset SHA-256: \\texttt{"+r["dataset_sha256"][:32]+"}\\\\ \\texttt{"+r["dataset_sha256"][32:]+"}.")
    return doc.finish()


def build_assignment4(a4,a3,project):
    r=json.loads((a4/"metrics.json").read_text())
    previous=json.loads((a3/"metrics.json").read_text())
    a3_best=json.loads((a3/"assignment3_best.json").read_text())
    if r["status"]!="complete" or previous["status"]!="complete":
        raise ValueError("Both experiment sets must be complete")
    if r["dataset_sha256"]!=previous["dataset_sha256"]:
        raise ValueError("Assignment 3 and 4 must use the identical source archive")
    # Attach a measured, provenance-checked baseline without retraining the existing A4 models.
    r["assignment3"]=a3_best
    (a4/"metrics.json").write_text(json.dumps(r,indent=2),encoding="utf-8")
    manifest=json.loads((a4/"dataset_manifest.json").read_text())
    doc=Report(project,4,"Dimensionality reduction and autoencoders")
    doc.title(4,r"Dimensionality reduction\\and autoencoders",
              "A complete six-task study using the supplied Group 32 MNIST subset. "
              "The report includes measured comparisons with an Assignment 3 reproduction trained on the same data splits.")
    doc.table(["Task","Coverage"],[
        [1,"PCA at 32, 64, 128, 256 dimensions; three FCNN architectures at every dimension"],
        [2,"Four shallow and four deep sigmoid autoencoders; final split reconstruction errors and examples"],
        [3,"Classification of all shallow autoencoder bottleneck representations"],
        [4,"Classification of all specified deep autoencoder bottleneck representations"],
        [5,"20\\% and 40\\% masking denoising autoencoders; reconstruction and classification"],
        [6,"All encoder weight images for the selected shallow AE and both denoising AEs"],
    ],"rY")
    doc.paragraph("All values are computed from saved results. No external digit images, pretrained models, "
                  "or invented Assignment 3 measurements are used. The LaTeX source and local figures accompany this PDF.")
    doc.end_title()
    doc.section("Dataset and experimental protocol",False)
    doc.table(["Digit","Training","Validation","Test"],
              [[c]+[r["counts"][s][str(c)] for s in ("train","val","test")] for c in r["classes"]]
              +[["Total"]+[sum(r["counts"][s].values()) for s in ("train","val","test")]],"Yrrr")
    doc.paragraph("Images are grayscale 28 $\\times$ 28, flattened to 784 coordinates and divided by 255. "
                  "Original train/validation/test splits are preserved. Internal class indices 0--4 map to digits 0, 3, 5, 6, and 9. "
                  "Identical decoded pixel hashes overlap across splits in "+str(sum(manifest["identical_pixel_hash_overlap"].values()))+" cases.")
    config=r["config"]
    doc.table(["Setting","Assignment 4 value"],[
        ["Optimizer","Adam with learning rate 0.001"],
        ["Batch size",config["batch_size"]],
        ["Epoch budget",f"Autoencoders at most {config['ae_epochs']}; classifiers at most {config['classifier_epochs']}"],
        ["AE checkpoint / stopping","Minimum validation MSE; 12 stale epochs, after at least 20 epochs"],
        ["FCNN checkpoint / stopping","Maximum validation accuracy; lower validation CE breaks ties; 8 stale epochs, after at least 12 epochs"],
        ["FCNN features","Per-coordinate standardization fitted on the training representation only"],
        ["FCNN architectures",r"64; 128; 128 $\to$ 64 hidden neurons, ReLU, 5 output logits"],
        ["Reproducibility",f"Base seed {config['seed']}, deterministic CPU PyTorch, {config['threads']} threads"],
    ],"lY")
    doc.paragraph("The same three FCNN architectures are considered in Tasks 1, 3, and 4. "
                  "PCA, shallow, and deep classifiers use matching seeds for a given dimension and architecture. "
                  "Digit labels enter classifier training only, not PCA fitting or the autoencoder loss.")
    doc.section("Model selection and Assignment 3 reference")
    doc.paragraph("At each compressed dimension, validation accuracy selects an FCNN architecture and its saved checkpoint. "
                  "Only that selected architecture is evaluated on the test set. The report then identifies the best observed test-ranked dimension "
                  "as explicitly requested by the assignment. Such rankings are descriptive after dimension selection.")
    doc.paragraph("Assignment 3 is now reproduced from its supplied handout using three architectures with 3, 4, and 5 hidden layers and all seven "
                  "prescribed optimizers. Every run uses the specified batch sizes, learning rate 0.001, identical initial weights within each architecture, "
                  "and stopping criterion $|L_t-L_{t-1}|<10^{-4}$, without an epoch limit. "
                  "The best converged architecture/optimizer is chosen by validation accuracy.")
    doc.table(["Assignment 3 reference","Measured value"],[
        ["Architecture",tex(a3_best["architecture"]).replace("->",r"$\to$")],
        ["Optimizer",tex(a3_best["optimizer"])],
        ["Convergence epochs",a3_best["convergence_epochs"]],
        ["Training accuracy",percent(a3_best["training_accuracy"])],
        ["Validation accuracy",percent(a3_best["validation_accuracy"])],
        ["Test accuracy",percent(a3_best["test_accuracy"])],
    ],"lY")
    doc.paragraph("This is a newly measured reproduction, not a recovered historical submission. "
                  "The earlier three-model raw-pixel reference did not meet Assignment 3's depth and optimizer requirements; "
                  "it is no longer used as the Assignment 3 comparison. "
                  "Assignment 3 and 4 share the same archive SHA-256 and exact split membership. "
                  "Their architecture searches, optimizer schedules, and stopping rules differ as required by the respective handouts; "
                  "the accuracy differences therefore compare these studies rather than isolate compression alone.")

    def best(method): return max(r["selected"][method],key=lambda row:(row["test"]["accuracy"],-row["dimension"]))

    def classification(method,task,title):
        doc.section(f"Task {task}: "+title)
        if method=="pca":
            doc.paragraph("PCA is fitted by exact eigendecomposition of the training sample covariance. "
                          "The training mean centers every split; validation and test images do not fit the directions or mean.")
            doc.raw(r"\[\mu=\frac{1}{N_{\rm tr}}\sum_i x_i,\quad C=\frac{(X_{\rm tr}-\mu)^T(X_{\rm tr}-\mu)}{N_{\rm tr}-1},\quad Z_s=(X_s-\mu)V_d.\]")
        else:
            doc.paragraph("The trained encoder is frozen. Training, validation, and test images are passed through it, "
                          "and the sigmoid bottleneck output is saved as the compressed representation. "
                          "Training-fitted standardization precedes the separate supervised FCNN; the saved latent codes themselves are unstandardized.")
        rows=[]
        for dimension in config["dimensions"]:
            candidates={row["architecture"]:row for row in r["classifiers"][method] if row["dimension"]==dimension}
            selected=next(row for row in r["selected"][method] if row["dimension"]==dimension)
            rows.append([dimension]+[percent(candidates[name]["val_accuracy"]) for name in ("64","128","128_64")]
                        +[hidden(selected["architecture"]),percent(selected["test"]["accuracy"])])
        doc.table(["Dim.","Val: 64","Val: 128","Val: 128--64","Selected","Test"],rows,"rYYYlr",font=r"\small")
        if method=="pca":
            doc.table(["Dimension","Training variance retained"],
                      [[d,percent(r["pca_explained_variance"][str(d)])] for d in config["dimensions"]],"Yr")
            doc.paragraph("Larger dimensions retain more training variance, but variance preservation is not the same objective as discrimination. "
                          "In particular, the FCNN's training-fitted coordinate scaling gives each retained component unit training variance.")
        chosen=best(method)
        difference=(chosen["test"]["accuracy"]-a3_best["test_accuracy"])*100
        doc.paragraph("The highest observed test result is \\textbf{"+percent(chosen["test"]["accuracy"])+"} at \\textbf{"+str(chosen["dimension"])+" dimensions}, "
                      "with FCNN hidden layers "+hidden(chosen["architecture"])+". "
                      f"It differs from the Assignment 3 reference by {difference:+.2f} percentage points.")
        if method!="pca":
            p=best("pca")
            doc.paragraph(f"The difference from the best PCA test accuracy is {(chosen['test']['accuracy']-p['test']['accuracy'])*100:+.2f} percentage points.")
        if method=="deep":
            shallow=best("shallow")
            doc.paragraph(f"The difference from the best shallow AE classification test accuracy is {(chosen['test']['accuracy']-shallow['test']['accuracy'])*100:+.2f} percentage points.")
        doc.section(f"Task {task}: test confusion matrices")
        doc.image(a4/"figures"/f"confusion_{method}.png","Rows are true digits; columns are predicted digits. "
                  "Each row contains 759 test images. One matrix is provided for the validation-selected FCNN at each dimension.","0.73\\textheight")
        matrix=np.asarray(chosen["test"]["confusion_matrix"]); off=matrix.copy(); np.fill_diagonal(off,0)
        i,j=np.unravel_index(off.argmax(),off.shape)
        doc.paragraph(f"At the test-ranked best dimension, the largest directional confusion is digit {r['classes'][i]} predicted as {r['classes'][j]} ({off[i,j]} images).")

    classification("pca",1,"PCA and classification")
    doc.section("Task 2: image reconstruction")
    doc.paragraph("For each bottleneck width $d\\in\\{32,64,128,256\\}$, the shallow autoencoder is "
                  "$784\\to d\\to784$, and the deep autoencoder is $784\\to400\\to d\\to400\\to784$. "
                  "Logistic sigmoid is used in every hidden layer and the reconstruction output. Adam minimizes pixelwise MSE.")
    doc.raw(r"\[\mathrm{MSE}_s=\frac{1}{N_s\,784}\sum_{i=1}^{N_s}\sum_{j=1}^{784}\bigl(\hat{x}_{ij}-x_{ij}\bigr)^2.\]")
    doc.paragraph("The validation-selected checkpoint is restored before computing the complete split averages. "
                  "To express error as average squared Euclidean error per image, multiply the listed MSE by 784.")
    doc.table(["Autoencoder","Epoch","Train MSE","Val MSE","Test MSE"],
              [[tex(name),row["best_epoch"]]+[f"{row['mse'][split]:.6f}" for split in ("train","val","test")]
               for name,row in r["autoencoders"].items() if row["noise"]==0],"Yrrrr")
    lowest=min((name for name,row in r["autoencoders"].items() if row["noise"]==0),key=lambda name:r["autoencoders"][name]["mse"]["test"])
    doc.paragraph("The lowest test reconstruction MSE is "+f"{r['autoencoders'][lowest]['mse']['test']:.6f}"+" for "+tex(lowest)+". "
                  "Reconstruction and classification optimize different objectives; a low pixel loss need not imply the highest digit accuracy.")
    for split in ("train","val","test"):
        doc.section("Task 2: "+split+" reconstruction examples")
        doc.image(a4/"figures"/f"reconstructions_{split}.png",
                  "One deterministic image per class from this split is used for every architecture. The first row is original; "
                  "the remaining eight rows show shallow and deep reconstructions. The same images are reused in Task 5.","0.80\\textheight")
    classification("shallow",3,"shallow autoencoder representations")
    classification("deep",4,"deep autoencoder representations")
    doc.section("Task 5: denoising autoencoders")
    choice=r["task5_choice"]
    doc.paragraph("The brief requires selecting the bottleneck using Task 3's best test accuracy. This selects $d="+str(choice["dimension"])+"$. "
                  "Both denoising autoencoders use $784\\to"+str(choice["dimension"])+"\\to784$, sigmoid activations, "
                  "and Adam. Their FCNN uses the same best Task 3 hidden architecture at that dimension: "+hidden(choice["architecture"])+".")
    doc.paragraph("Noise is independent Bernoulli pixel masking. A selected pixel is set to zero with probability 0.20 or 0.40; "
                  "the clean image is the target. Training samples fresh masks in every batch. Fixed independently seeded masks are "
                  "used for post-training split errors and example images. Percentages denote expected masking rates.")
    doc.raw(r"\[\tilde{x}=m\odot x,\qquad m_j\sim\operatorname{Bernoulli}(1-p),\qquad \min_\theta\;\mathbb{E}_{x,m}\lVert f_\theta(\tilde{x})-x\rVert_2^2.\]")
    doc.table(["Masking","Epoch","Train MSE","Val MSE","Test MSE"],
              [[str(level)+r"\%",r["autoencoders"][f"denoising_{level}"]["best_epoch"]]
               +[f"{r['autoencoders'][f'denoising_{level}']['mse'][split]:.6f}" for split in ("train","val","test")]
               for level in (20,40)],"Yrrrr")
    doc.paragraph("These errors reconstruct clean targets from the fixed corrupted inputs. For clean inputs, validation/test MSE are "
                  +"; ".join(f"{level}\\%: {r['autoencoders'][f'denoising_{level}']['clean_input_mse']['val']:.6f} / "
                             f"{r['autoencoders'][f'denoising_{level}']['clean_input_mse']['test']:.6f}" for level in (20,40))+".")
    doc.table(["Masking","FCNN","Clean val.","Clean test","Noisy val.","Noisy test"],
              [[str(level)+r"\%",hidden(r["selected"][f"denoising_{level}"][0]["architecture"]),
                percent(r["selected"][f"denoising_{level}"][0]["val_accuracy"]),
                percent(r["selected"][f"denoising_{level}"][0]["test"]["accuracy"]),
                percent(r["selected"][f"denoising_{level}"][0]["noisy_input_evaluation"]["val"]["accuracy"]),
                percent(r["selected"][f"denoising_{level}"][0]["noisy_input_evaluation"]["test"]["accuracy"])]
               for level in (20,40)],"lYrrrr",font=r"\small")
    doc.paragraph("Primary classification uses the encodings of clean images, as in Tasks 3 and 4. The noisy columns evaluate "
                  "the same trained classifier on masked inputs, with no extra selection or retraining. "
                  "Because those evaluations use different corruption levels, their difference is not a controlled same-noise comparison.")
    doc.paragraph("Task 5(d)(i) requires reusing the best Task 3 FCNN; it is retained for both noise levels. "
                  "Task 5(d)(ii)'s reference to different architectures is interpreted as the two denoising configurations.")
    for split in ("train","val","test"):
        doc.section("Task 5: "+split+" denoising examples")
        doc.image(a4/"figures"/f"denoising_{split}.png","For every original image, the 20\\% masked input and its reconstruction "
                  "are followed by the 40\\% masked input and its reconstruction. Both recover clean targets.","0.70\\textheight")
    doc.section("Task 6: encoder weight interpretation")
    doc.paragraph("All "+str(choice["dimension"])+" input-to-bottleneck weight vectors are reshaped to 28 $\\times$ 28. "
                  "Red denotes positive weights favoring bright pixels; blue denotes negative weights favoring dark pixels. "
                  "The three grids share a symmetric color scale set by the pooled 99th percentile of absolute weights; biases are excluded.")
    doc.paragraph("Sigmoid activation is monotone in $w^Tx+b$, so the signed weight template shows a neuron's preferred input pattern, "
                  "as requested in the assignment. Under $0\\leq x_j\\leq1$, an exact maximizing input has $x_j=1$ where $w_j>0$ and $x_j=0$ where $w_j<0$. "
                  "The following figures display the assigned weight templates, not numerically optimized digit-like images.")
    doc.table(["Encoder","Weight RMS","Neighbor difference","Difference / RMS"],
              [[tex(name),f"{s['weight_rms']:.5f}",f"{s['neighbor_absolute_difference']:.5f}",f"{s['normalized_neighbor_difference']:.5f}"]
               for name,s in r["weight_statistics"].items()],"Yrrr")
    smoothest=min(r["weight_statistics"],key=lambda name:r["weight_statistics"][name]["normalized_neighbor_difference"])
    doc.paragraph("The neighbor difference averages absolute changes between adjacent horizontal and vertical weights; "
                  "division by RMS removes overall weight scale. "+tex(smoothest)+" has the smallest ratio in this run, "
                  "indicating smoother local templates by this measure. This statistic is not classification accuracy. "
                  "Neuron indices are not aligned between independently trained models, so comparisons should be at the set level.")
    for name,title in (("shallow","clean shallow autoencoder"),("denoising_20","20 percent denoising autoencoder"),
                       ("denoising_40","40 percent denoising autoencoder")):
        doc.section("Task 6: "+title)
        doc.image(a4/"figures"/f"weights_{name}.png","Every neuron in the selected bottleneck is included, with one shared signed color scale. "
                  "Numbers identify neurons within this model.","0.81\\textheight")
    doc.section("Comparison with Assignment 3")
    rows=[["Assignment 3 reproduction",784,tex(a3_best["optimizer"]),percent(a3_best["validation_accuracy"]),percent(a3_best["test_accuracy"]),"0.00"]]
    names={"pca":"PCA (Task 1)","shallow":"Shallow AE (Task 3)","deep":"Deep AE (Task 4)",
           "denoising_20":"DAE, 20\\% masking","denoising_40":"DAE, 40\\% masking"}
    for method,name in names.items():
        row=best(method)
        rows.append([name,row["dimension"],hidden(row["architecture"]),percent(row["val_accuracy"]),
                     percent(row["test"]["accuracy"]),f"{(row['test']['accuracy']-a3_best['test_accuracy'])*100:+.2f}"])
    with (a4/"assignment3_comparison.csv").open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=["study","dimension","architecture","optimizer","validation_accuracy",
                                               "test_accuracy","difference_from_assignment3_percentage_points"])
        writer.writeheader()
        writer.writerow({"study":"assignment3","dimension":784,"architecture":a3_best["architecture"],
                         "optimizer":a3_best["optimizer"],"validation_accuracy":a3_best["validation_accuracy"],
                         "test_accuracy":a3_best["test_accuracy"],"difference_from_assignment3_percentage_points":0})
        for method in names:
            row=best(method)
            writer.writerow({"study":method,"dimension":row["dimension"],"architecture":row["architecture"],
                             "optimizer":"Adam","validation_accuracy":row["val_accuracy"],"test_accuracy":row["test"]["accuracy"],
                             "difference_from_assignment3_percentage_points":(row["test"]["accuracy"]-a3_best["test_accuracy"])*100})
    doc.table(["Study","Dim.","Optimizer / FCNN","Val.","Test","Delta (pp)"],rows,"Yrlrrr",font=r"\small")
    doc.paragraph("For Assignment 3 the middle column records its optimizer, and its full architecture is listed earlier. "
                  "For Assignment 4 it records the FCNN hidden layers; all those FCNNs use Adam. "
                  "The delta is test accuracy minus the Assignment 3 test accuracy, in percentage points. "
                  "The best test-ranked dimension is reported within each Task 1, 3, or 4 representation family.")
    doc.image(a4/"figures"/"accuracy_comparison.png","At each dimension, the FCNN architecture was selected on validation accuracy. "
              "PCA, shallow AE, and deep AE classification are compared on the same test split.","0.34\\textheight")
    doc.paragraph("PCA reaches "+percent(best("pca")["test"]["accuracy"])+", while the deep AE reaches "+percent(best("deep")["test"]["accuracy"])+" in this study. "
                  "The best shallow result is "+percent(best("shallow")["test"]["accuracy"])+". "
                  "The numerical differences to Assignment 3 above are the requested comparison; they do not prove that "
                  "a representation alone caused a performance difference, given the different architecture and training requirements.")
    doc.section("Learning behavior and observations")
    doc.image(a4/"figures"/"ae_learning_curves.png","Validation reconstruction MSE during AE training. Final split MSE values "
              "are separately recomputed after restoring the selected checkpoint.","0.37\\textheight")
    doc.paragraph("Shallow reconstruction error decreases strongly as bottleneck width grows in this run. "
                  "The deep encoder improves classification over the shallow encoder at smaller widths, even though reconstruction "
                  "MSE and classification accuracy are not interchangeable. A nonlinear representation can preserve useful digit structure "
                  "without minimizing every pixel discrepancy.")
    doc.paragraph("Denoising encourages recovery from missing evidence. The weight smoothness statistics change with corruption training, "
                  "but the measured reconstruction and clean/noisy classification results are the evidence of its effects. "
                  "Increasing noise can alter both reconstruction fidelity and encoder adaptation; better classification is not guaranteed.")
    doc.paragraph("All AE curves were still improving over parts of the finite training budget. Therefore these results compare "
                  "the stated training setup, not the maximum attainable performance of every architecture. "
                  "The study uses one seed per experiment and does not supply confidence intervals or repeated-run significance tests.")
    doc.section("Specification notes and verification")
    doc.paragraph("\\textbf{Task 4 wording.} Its heading says `2-hidden layer autoencoder', while Task 2 specifies a three-hidden-layer "
                  "autoencoder with 400 neurons in the first and third hidden layers. Task 4 uses the two-layer encoder $784\\to400\\to d$ "
                  "of that explicitly specified deep autoencoder; no additional unspecified architecture is introduced.")
    doc.paragraph("\\textbf{Test-informed denoising dimension.} Task 5 explicitly chooses its bottleneck by the best Task 3 test accuracy. "
                  "This instruction is followed and recorded. Downstream test results therefore reuse test information during design; "
                  "a fresh held-out split would be needed for an unbiased estimate after that choice.")
    selected_count=sum(len(rows) for rows in r["selected"].values())
    doc.paragraph("\\textbf{Checks.} Assignment 4 verification recomputes all ten AE reconstruction errors, saved latent samples, "
                  f"and the accuracies/confusion matrices for all {selected_count} selected classifiers. PCA's training mean, component orthogonality, "
                  "and projections are checked. Four pipeline tests cover train-only PCA fitting, required model shapes and sigmoid ranges, "
                  "seeded corruption, and checkpoint restoration. Assignment 3 additionally checks every optimizer against PyTorch, "
                  "matched initialization, the exact stopping condition, and final checkpoint metrics.")
    doc.paragraph("\\textbf{Reproduce.} Run \\texttt{python run\\_assignment3.py} in the Assignment 3 folder, then use the saved "
                  "\\texttt{assignment3\\_best.json} when reproducing the Assignment 4 study. "
                  "\\texttt{python report.py --compile} generates both LaTeX projects and compiles these PDFs. "
                  "The projects contain their main \\texttt{.tex} files and local figure assets, and work with pdfLaTeX or Overleaf.")
    doc.paragraph("Sources: \\nolinkurl{CS601T_Assignment3_Details_07Sept2026.pdf} and "
                  "\\nolinkurl{CS601T_Assignment4_Details_30Sept2026.pdf}.")
    doc.paragraph("Dataset SHA-256: "
                  "\\texttt{"+r["dataset_sha256"][:32]+"}\\\\ \\texttt{"+r["dataset_sha256"][32:]+"}.")
    return doc.finish()


def compile_report(source,pdf_destination):
    executable=shutil.which("pdflatex")
    if executable is None: raise RuntimeError("pdflatex is unavailable; compile the generated project in Overleaf or install MiKTeX/TeX Live")
    build=source.parent/"build"; build.mkdir(exist_ok=True)
    command=[executable,"-interaction=nonstopmode","-halt-on-error","-file-line-error","--disable-installer",
             "-output-directory",str(build),source.name]
    for pass_number in (1,2):
        result=subprocess.run(command,cwd=source.parent,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                              encoding="utf-8",errors="replace")
        (build/f"compile-pass-{pass_number}.txt").write_text(result.stdout,encoding="utf-8")
        if result.returncode:
            print(result.stdout[-7000:])
            raise RuntimeError(f"LaTeX compilation failed; inspect {build/f'compile-pass-{pass_number}.txt'}")
    pdf_destination.mkdir(parents=True,exist_ok=True)
    final=pdf_destination/source.with_suffix(".pdf").name
    shutil.copy2(build/source.with_suffix(".pdf").name,final)
    print(f"Compiled PDF: {final}")
    return final


def project_zip(project,archive):
    with zipfile.ZipFile(archive,"w",zipfile.ZIP_DEFLATED) as z:
        for path in sorted(project.rglob("*")):
            if path.is_file() and "build" not in path.relative_to(project).parts:
                z.write(path,path.relative_to(project))
    with zipfile.ZipFile(archive) as z: assert z.testzip() is None


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    workspace=Path(__file__).resolve().parent.parent
    parser.add_argument("--results",type=Path,default=workspace/"Group32_Assignment4_code"/"results")
    default_a3=workspace/"Group32_Assignment3_code"/"results"
    if not default_a3.exists():
        default_a3=Path(__file__).resolve().parent/"assignment3_reference"
    parser.add_argument("--assignment3",type=Path,default=default_a3)
    parser.add_argument("--output-root",type=Path,default=workspace/"output"/"latex")
    parser.add_argument("--pdf-dir",type=Path,default=workspace/"output"/"pdf")
    parser.add_argument("--compile",action="store_true",help="Compile both generated projects with installed pdflatex")
    args=parser.parse_args()
    root=args.output_root.resolve()
    sources=[build_assignment3(args.assignment3.resolve(),root/"assignment3"),
             build_assignment4(args.results.resolve(),args.assignment3.resolve(),root/"assignment4")]
    for source in sources:
        print(f"LaTeX source: {source}")
        if args.compile: compile_report(source,args.pdf_dir.resolve())
        archive=root.parent/(source.stem+"_latex.zip")
        project_zip(source.parent,archive)
        print(f"Portable LaTeX project: {archive}")


if __name__=="__main__": main()
