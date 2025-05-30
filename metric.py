import os
import time
import random

import numpy as np
import regex as re
from claudette import Chat, models
from evaluate import load
from anthropic import RateLimitError
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class Metric:
    def __init__(self, **kwargs):
        self._load_metric(**kwargs)

    def _load_metric(self, **kwargs):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def compute(self, prompts, predictions, references):
        raise NotImplementedError("This method should be overridden by subclasses.")


class Rouge(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _load_metric(self, **kwargs):
        self.metric = load("rouge", keep_in_memory=True)

    def compute(self, prompts, predictions, references):
        return self.metric.compute(predictions=predictions, references=references)


class Bleurt(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _load_metric(self, **kwargs):
        self.metric = load("bleurt", keep_in_memory=True)

    def compute(self, prompts, predictions, references):
        return np.mean(
            self.metric.compute(predictions=predictions, references=references)[
                "scores"
            ]
        )


class BertScore(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _load_metric(self, **kwargs):
        self.metric = load("bertscore", keep_in_memory=True)

    def compute(self, prompts, predictions, references):
        result = self.metric.compute(
            predictions=predictions, references=references, lang="en"
        )
        return {
            "precision": np.mean(result["precision"]),
            "recall": np.mean(result["recall"]),
            "f1": np.mean(result["f1"]),
        }


class Accuracy(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _load_metric(self, **kwargs):
        from sklearn.metrics import accuracy_score

        self.metric = accuracy_score

    def compute(self, prompts, predictions, references):
        return self.metric(references, predictions)


class ExactMatchScore(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _load_metric(self, **kwargs):
        pass

    def compute(self, prompts, predictions, references):
        return np.mean(
            [
                1 if p.split() == r.split() else 0
                for p, r in zip(predictions, references)
            ]
        )


class LevenshteinDistance(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _load_metric(self, **kwargs):
        from fuzzywuzzy import fuzz

        self.metric = fuzz.ratio

    def compute(self, prompts, predictions, references):
        return np.mean([self.metric(p, r) for p, r in zip(predictions, references)])


class RulerStringMatch(Metric):
    """
    Metric used in RULER.
    Reference: https://github.com/hsiehjackson/RULER/blob/main/scripts/eval/synthetic/constants.py
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def postprocess_pred(predict_str: str):
        predict_str = predict_str.strip()

        # Remove all non-printable characters
        np_pattern = re.compile(r"[\x00-\x1f]")
        predict_str = np_pattern.sub("\n", predict_str).strip()

        return predict_str

    @staticmethod
    def string_match_part(refs, preds):
        scores = [
            max([1.0 if r.lower() in pred.lower() else 0.0 for r in ref])
            for pred, ref in zip(preds, refs)
        ]
        score = sum(scores) / len(preds) * 100
        return {"score": round(score, 4)}

    @staticmethod
    def string_match_all(refs, preds):
        scores = [
            sum([1.0 if r.lower() in pred.lower() else 0.0 for r in ref]) / len(ref)
            for pred, ref in zip(preds, refs)
        ]
        score = sum(scores) / len(preds) * 100
        return {"score": round(score, 4)}

    def _load_metric(self, **kwargs):
        if kwargs.get("match_part", False):
            self.metric = self.string_match_part
        else:
            self.metric = self.string_match_all

    def compute(self, prompts, predictions, references):
        predictions = [self.postprocess_pred(pred) for pred in predictions]
        return self.metric(references, predictions)


REFERENCE_TEMPLATE = """You are shown ground-truth answer(s) and asked to judge the quality of an LLM-generated answer.
Assign it a score from 0-9 where 0 is the worst and 9 is the best based on how similar it is to the ground-truth(s).
Do NOT explain your choice. Simply return a number from 0-9.

====GROUND TRUTHS====
{labels}

====ANSWER====
{prediction}"""

PREFILL = "====RESULT====\nThe score (0-9) is:"


class LLMRouge(Metric):
    def __init__(self, num_retries=5, **kwargs) -> None:
        assert (
            "ANTHROPIC_API_KEY" in os.environ
        ), "Please set the ANTHROPIC_API_KEY environment variable."
        super().__init__(**kwargs)
        self.num_retries = num_retries

    def _load_metric(self, **kwargs):
        name = kwargs.get(
            "name", "claude-3-5-haiku-20241022"
        )  # haiku got a new version, specified now "claude-3-5-haiku-20241022"
        matching_names = [m for m in models if name in m]
        assert len(matching_names) > 0, f"Model name {name} not found in {models}"
        assert (
            len(matching_names) == 1
        ), f"Model name {name} found x{len(matching_names)} in {models}"
        self.chat = Chat(
            matching_names[0], sp="""You are a helpful and concise assistant."""
        )

    def parse_int(self, text):
        return int(re.search(r"\d+", text).group())

    def compute(self, prompts, predictions, labels):
        scores = []
        for p, ls in zip(predictions, labels):
            prompt = REFERENCE_TEMPLATE.format(labels="\n---\n".join(ls), prediction=p)
            # Clear conversation history
            self.chat.h = []
            try:
                score = (
                    self.chat(prompt, prefill=PREFILL)
                    .content[0]
                    .text[len(PREFILL) :]
                    .strip()
                )
            except RateLimitError:
                retries = 0
                while retries < self.num_retries:
                    time.sleep(10)
                    try:
                        score = (
                            self.chat(prompt, prefill=PREFILL)
                            .content[0]
                            .text[len(PREFILL) :]
                            .strip()
                        )
                        break
                    except RateLimitError:
                        retries += 1
                if retries == self.num_retries:
                    raise RateLimitError("Exceeded maximum number of retries.")

            score = self.parse_int(score)
            scores.append(score)
        return {"llm_rouge": sum(scores) / len(scores)}


# NOTE: Experimental feature, not yet fully functional
class LLMRougeLlama(Metric):
    """
    LLM-based ROUGE evaluator using Meta's LLaMA 3.2 3B Instruct model.

    This metric queries the LLM with a templated prompt comparing a prediction to
    reference labels. The LLM returns a numeric ROUGE-like score, extracted from its output.
    """

    def __init__(
        self,
        num_retries=5,
        **kwargs,
    ):
        """
        Initializes the metric.

        Args:
            num_retries (int): Maximum number of retry attempts if generation fails.
            **kwargs: Additional arguments for the parent Metric class.
        """
        super().__init__(**kwargs)
        self.num_retries = num_retries

    def _load_metric(self, **kwargs):
        """
        Loads the LLaMA model and tokenizer.
        """
        model_name = "meta-llama/Llama-3.2-3B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="auto"
        )
        self.model.eval()

    def parse_int(self, text):
        """
        Extracts the first digit from the model output.

        Args:
            text (str): The LLM-generated output text.

        Returns:
            int: The first digit found in the text.

        Raises:
            ValueError: If no digit is found.
        """
        return int(re.search(r"\d", text).group())

    def _generate_response(self, prompt, max_new_tokens=100):
        """
        Generates a response from the LLM given a prompt.

        Args:
            prompt (str): The prompt to feed into the LLM.
            max_new_tokens (int): Maximum number of new tokens to generate.

        Returns:
            str: Decoded LLM output text (stripped).
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=(
                    self.tokenizer.eos_token_id
                    if self.tokenizer.eos_token_id
                    else self.tokenizer.pad_token_id
                ),
            )
        decoded = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True
        )
        return decoded.strip()

    def compute(self, prompts, predictions, labels):
        """
        Computes average LLM-based ROUGE scores over all samples.

        For each prediction-label pair, a reference prompt is constructed and scored by the LLM.
        If generation fails, retries are attempted with exponential backoff.

        Args:
            prompts: Unused (for compatibility with base Metric).
            predictions (List[str]): Model-generated outputs.
            labels (List[str]): Reference outputs.

        Returns:
            Dict[str, float]: Dictionary with averaged LLM-based ROUGE score.
        """
        scores = []

        for p, ls in zip(predictions, labels):
            prompt = REFERENCE_TEMPLATE.format(labels=ls, prediction=p) + PREFILL

            retries = 0
            while retries <= self.num_retries:
                try:
                    response = self._generate_response(prompt)
                    score = self.parse_int(response)
                    scores.append(score)
                    break
                except Exception as e:
                    retries += 1
                    if retries > self.num_retries:
                        raise RuntimeError(f"Max retries reached. Last error: {e}")
                    time.sleep(10)

        return {"llm_rouge": np.mean(scores)}


LLM_JUDGE_TEMPLATE = """You are shown a prompt and asked to assess the quality of an LLM-generated answer on the following dimensions:

===CRITERIA===
{criteria}

Assign a score from 0-9 where 0 is the worst and 9 is the best based on how well the answer meets the criteria. DO NOT explain your choice or add anything else to your answer. Example: "helpful: 8\ncoherent: 9\nfaithful: 7".

====PROMPT====
{prompt}

====ANSWER====
{prediction}"""


CRITERIA = {
    "helpful": "The answer executes the action requested by the prompt without extraneous detail.",
    "coherent": "The answer is logically structured and coherent (ignore the prompt).",
    "faithful": "The answer is faithful to the prompt and does not contain false information.",
}


class LLMJudge(LLMRouge):
    def __init__(self, **kwargs) -> None:
        assert (
            "ANTHROPIC_API_KEY" in os.environ
        ), "Please set the ANTHROPIC_API_KEY environment variable."
        super().__init__(**kwargs)

        self.criteria = list(sorted([k for k in CRITERIA]))
        self.criteria_def = "\n".join([f"{k}: {CRITERIA[k]}" for k in self.criteria])
        self.prefill = (
            f"\n\n====SCORES for {', '.join(self.criteria)}====\n\n{self.criteria[0]}:"
        )

    def parse_scorecard(self, scorecard):
        try:
            return {
                k: int(v)
                for k, v in dict(
                    re.findall(rf"({'|'.join(self.criteria)})\W+(\d+)", scorecard)
                ).items()
            }
        except Exception as e:
            print(e)
            raise Exception(
                f"Could not parse LLM-generated scorecard for {self.__class__}:\n{scorecard}"
            )

    def claudette_scorecard(self, prompt, prediction):
        prompt = LLM_JUDGE_TEMPLATE.format(
            criteria=self.criteria_def, prompt=prompt, prediction=prediction
        )
        # Clear conversation history
        self.chat.h = []
        scorecard = (
            self.chat(prompt, prefill=self.prefill)
            .content[0]
            .text[len(self.prefill) - len(self.criteria[0]) - 1 :]
            .strip()
        )
        return scorecard

    def compute(self, prompts, predictions, labels):
        scores = []

        for prompt, pred in zip(prompts, predictions):
            scorecard = self.claudette_scorecard(prompt, pred)
            score_dict = self.parse_scorecard(scorecard)
            scores.append(score_dict)

        return {k: np.mean([s[k] for s in scores]) for k in self.criteria}


# NOTE: Experimental feature, not yet fully functional
class LLMJudgeLlama(LLMRougeLlama):
    """
    LLM-based multi-criteria evaluator using Meta's LLaMA 3.2 3B Instruct model.

    This metric prompts the LLM to score predictions across multiple evaluation
    criteria (e.g., coherence, faithfulness, helpfulness) and parses the
    generated scorecard into a dictionary of numeric scores.

    Attributes:
        criteria (List[str]): Sorted list of evaluation dimensions.
        criteria_def (str): Prompt-formatted description of each criterion.
        prefill (str): Scorecard header added to the prompt.
        max_new_tokens (int): Max token budget for scorecard generation.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.criteria = list(sorted([k for k in CRITERIA]))
        self.criteria_def = "\n".join([f"{k}: {CRITERIA[k]}" for k in self.criteria])
        self.prefill = f"\n\n====SCORES for {', '.join(self.criteria)}===="
        self.max_new_tokens = (
            len(self.tokenizer("helpful: 8\ncoherent: 9\nfaithful: 7")["input_ids"]) + 5
        )

    def parse_scorecard(self, scorecard):
        """
        Parses the LLM-generated scorecard into a dictionary of scores.

        Args:
            scorecard (str): Raw LLM output containing scores.

        Returns:
            Dict[str, int]: Dictionary mapping each criterion to its numeric score.

        Raises:
            Exception: If the scorecard cannot be parsed.
        """
        try:
            scores = {}
            for crit in self.criteria:
                # Match the criterion and the next number (score 1–5) after it
                pattern = rf"{crit}.*?(\d)"
                match = re.search(pattern, scorecard, flags=re.IGNORECASE | re.DOTALL)
                scores[crit] = int(match.group(1))
            return scores
        except Exception as e:
            print(e)
            raise Exception(
                f"Could not parse LLM-generated scorecard for {self.__class__}:\n{scorecard}"
            )

    def llama_scorecard(self, prompt, prediction):
        """
        Formats and sends the prompt to the LLM for scoring.

        Args:
            prompt (str): Original task prompt or context.
            prediction (str): Model-generated response.

        Returns:
            str: LLM-generated scorecard as a string.
        """
        input_text = (
            LLM_JUDGE_TEMPLATE.format(
                criteria=self.criteria_def,
                prompt=prompt,
                prediction=prediction,
            )
            + self.prefill
        )

        return self._generate_response(input_text, max_new_tokens=self.max_new_tokens)

    def compute(self, prompts, predictions, labels):
        """
        Computes mean score per criterion over all prediction-prompt pairs.

        Args:
            prompts (List[str]): Input prompts.
            predictions (List[str]): Model-generated outputs.
            labels (unused): Not used, included for API compatibility.

        Returns:
            Dict[str, float]: Mean score per evaluation criterion.
        """
        scores = []

        for p, pred in zip(prompts, predictions):
            retries = 0
            while retries <= self.num_retries:
                try:
                    scorecard = self.llama_scorecard(p, pred)
                    score_dict = self.parse_scorecard(scorecard)
                    scores.append(score_dict)
                    break
                except Exception as e:
                    retries += 1
                    if retries > self.num_retries:
                        raise RuntimeError(f"Max retries reached. Last error: {e}")
                    time.sleep(10)

        return {k: np.mean([s[k] for s in scores]) for k in self.criteria}


METRIC_MAPPING = {
    "accuracy": Accuracy,
    "bertscore": BertScore,
    "bleurt": Bleurt,
    "exact_match": ExactMatchScore,
    "levenshtein": LevenshteinDistance,
    "llm-rouge": LLMRouge,
    "llm-as-a-judge": LLMJudge,
    "llm-rouge-llama": LLMRougeLlama,
    "llm-as-a-judge-llama": LLMJudgeLlama,
    "rouge": Rouge,
    "ruler-string-match": RulerStringMatch,
}


class AutoMetric:
    def __init__(self):
        raise EnvironmentError(
            "This class is designed to be instantiated only through the from_name method"
        )

    def from_name(metric_name, **kwargs):
        if metric_name not in METRIC_MAPPING:
            raise ValueError(f"Invalid metric name: {metric_name}")
        return METRIC_MAPPING[metric_name](**kwargs)


if __name__ == "__main__":
    metric = AutoMetric.from_name("llm-as-a-judge")
    predictions = [
        "The answer to 2x2 is 4.",
        "The answer to 2x2 is 5.",
    ]
    labels = [["4"], ["4"]]
    prompts = [
        "What is 2x2?",
        "What is 2x2?",
    ]
    print(metric.compute(prompts=prompts, predictions=predictions, labels=None))
