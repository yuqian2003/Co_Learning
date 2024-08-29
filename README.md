# Co-Learning: Code Learning for Multi-Agent reinforcement Collaborative Framework with Conversational Natural Language Interfaces


## Framework Overview
The Co-Learning framework consists of five key agents:
1. **Main Agent**: Supervises and interacts with users.
2. **Correction Agent**: Revises and corrects code.
3. **Interpretation Agent**: Explains programming logic to identify incorrect code.
4. **Test Agent**: Tests the corrected code.
5. **Annotation Agent**: Adds comments to the revised code for better user understanding.

These agents communicate through conversational interfaces, and the system employs Environmental Reinforcement Learning (E-RL) to self-improve and provide feedback to both the agents and human users. The Co-Learning framework utilizes models such as ERNIE, SparkDesk, and LLaMa for different agents, and it evaluates code correction performance using criteria such as passing probability tests, single loop computation time, and the number of required loops.

## Key Contributions
1. Development of a Multi-Agent framework using multiple LLMs for code error correction.
2. Evaluation of LLM performance using an original dataset containing 702 error codes.
3. Exploration of reinforcement learning in a multi-agent environment based on large language models.
4. Benchmarking against existing frameworks, demonstrating significant improvements in accuracy and operating speed.

## Workflow
The Co-Learning framework operates within an environment created by the Main Agent. The workflow is as follows:
- The **Correction Agent** uses a default LLM to make an initial correction to the input error code.
- The **Test Agent** then evaluates the corrected code using test samples. If the code passes all tests, it is sent to the **Annotation Agent** for final annotation and output. 
- If the code fails any test, it is passed to the **Interpretation Agent** for further analysis. This process is stored in memory as an E-RL prompt.
- The system then selects the appropriate Correction Agent using reinforcement learning and generates a new code based on the memorized data and interpretation, forming a loop until the code passes all tests.

The system is designed to enhance code error correction by mimicking human debugging processes and adapting in real-time to select the most appropriate LLM model based on E-RL.

## Getting Started
To get started with the Co-Learning framework, you can clone this repository and follow the setup instructions provided.

## Datasets


## Citation

If you use this code in your research, please cite the following paper:

```bibtex
@article{MAS4POI2024,
  title={MAS4POI: A Multi-Agent System Collaboration for Next POI Recommendation},
  author={Yuqian Wu, Yuhong Peng, Raymond S.T. Lee},
  journal={Journal Name},
  year={2024},
  url={https://github.com/yuqian2003/MAS4POI}
}
```
# Acknowledgments:
This research was supported by *Beijing Normal University-Hong Kong Baptist University United International College (UIC)* and the *IRADs lab*. 
The authors express their gratitude for providing the necessary computational facilities.
