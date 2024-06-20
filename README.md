# Alignment Analysis Utilizing Generative Pretrained Transformer 4 (GPT-4)
This repository contains the experimental resources employed in the Bachelor's thesis entitled "Assessing Corporate Communication Alignment with Stated Goals Using Large Language Models". This thesis, conducted at Leiden Institute of Advanced Computer Science (LIACS) under the supervision of  Dr. P.W.H. van der Putten (first supervisor) and Prof.dr.ir. J.M.W. Visser (second supervisor), was authored by Alexander Serraris.

# Experimental Procedures

## Procedure for the Conversion of Stated Goals into Principles
The stated goals of the company, extracted from the `stated goals.csv` file, were systematically transformed into principles. This transformation process involved the replacement of variable `[[X]]` with the stated goals in the `translation-into-principles.txt` file and sending the subsequent prompt to GPT-4. The output of this prompt was subsequently subjected to validation using the JSON schema delineated in the `translation-jsonschema.json` file.

## Procedure for the Execution of Alignment Analysis
The principles, derived from the aforementioned procedure, were analyzed on alignment with a self-published article from the company. This article was extracted from the `articles.csv` file for the respective company. The article, along with the stated goals, were input into the prompt delineated in the `analysis-scoring.txt` file, respectively replacing variables `[[X]]` and `[[principles]]` with their corresponding contents. The output of this analysis was subjected to validation using the JSON schema delineated in the `analysis-jsonschema.json` file.
