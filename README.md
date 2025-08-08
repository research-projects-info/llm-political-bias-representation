# LLMExperiments
From reference paper:
- 📄 Paper: [https://arxiv.org/abs/2502.16280](https://arxiv.org/abs/2502.16280)

For  "4.2 Constructing a probe for identifying party-related MLP value vectors" use [ProbeClassifier.py](./ProbeClassifier.py) script.

For "4.3 Analyzing the mapping between personas and the identified party-related value vectors" execute [PersonMapping.py](./PersonMapping.py) script. This uses de output of 4.2.

To submit a batch task to the cluster, use the [run_job_probe.sh](./run_job_probe.sh) script.