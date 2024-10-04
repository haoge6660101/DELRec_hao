# DELRec: Distilling Sequential Pattern to Enhance LLMs-based Sequential Recommendation
## Abstract
Sequential recommendation (SR) tasks aim to predict users’ next interaction by learning their behavior sequence  and capturing the connection between users’ past interactions  and their changing preferences. Conventional SR models often  focus solely on capturing sequential patterns within the training  data, neglecting the broader context and semantic information  embedded in item titles from external sources. This limits  their predictive power and adaptability. Recently, large language  models (LLMs) have shown promise in SR tasks due to their  advanced understanding capabilities and strong generalization  abilities. Researchers have attempted to enhance LLMs-based  recommendation performance by incorporating information from  conventional SR models. However, previous approaches have encountered problems such as 1) limited textual information leading  to poor recommendation performance; 2) incomplete understanding and utilization of conventional SR models information by LLMs; and 3) excessive complexity and low interpretability of LLMs-based methods.

To improve the performance of LLMs-based SR, we propose a novel framework, Distilling Sequential Pattern to Enhance LLMs-based Sequential Recommendation (DELRec), which aims  to extract knowledge from conventional SR models and enable LLMs to easily comprehend and utilize the extracted knowledge  for more effective SRs. DELRec consists of two main stages: 1) Distill Pattern from Conventional SR Models, focusing on  extracting behavioral patterns exhibited by conventional SR  models using soft prompts through two well-designed strategies; 2) LLMs-based Sequential Recommendation, aiming to fine-tune LLMs to effectively use the distilled auxiliary information to  perform SR tasks. Extensive experimental results conducted  on three real datasets validate the effectiveness of the DELRec framework.

## Rough View
![delrec_rough](https://github.com/user-attachments/assets/b61bf4fd-9775-4bd5-9e64-b23829873450)

## Paper
DELRec: Distilling Sequential Pattern to Enhance LLMs-based Sequential Recommendation([DELRec.pdf](https://github.com/user-attachments/files/16639272/DELRec.pdf))

## Preparation
1. **Prepare the environment:**
   To install the dependencies for this project, run the following command:
    ```bash
    git clone https://github.com/haoge6660101/DELRec_hao.git
    cd DELRec
    pip install -r requirements.txt
    ```

2. **Prepare the pre-trained Huggingface model of Flan-T5-XL(https://huggingface.co/google/flan-t5-xl).**

3. **Download the datasets.**
   MovieLens-100K:[](https://grouplens.org/datasets/movielens/100k/)
