import requests
from github import Github
from github.GithubException import GithubException
import pylint.lint
from flake8.api import legacy as flake8
import redis
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from typing import List, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AutomatedCodeReview:
    def __init__(self, github_token: str, redis_url: str):
        self.github = Github(github_token)
        self.redis = redis.from_url(redis_url, decode_responses=True)
        self.model, self.tokenizer = self.load_model()

    @staticmethod
    def load_model():
        model_name = "microsoft/codebert-base"
        tokenizer = RobertaTokenizer.from_pretrained(model_name)
        model = RobertaForSequenceClassification.from_pretrained(model_name)
        return model, tokenizer

    def analyze_pull_request(self, repo_name: str, pr_number: int):
        try:
            repo = self.github.get_repo(repo_name)
            pr = repo.get_pull(pr_number)
        except GithubException as e:
            logger.error(f"Error accessing GitHub: {e}")
            return

        results = []
        for file in pr.get_files():
            if file.filename.endswith('.py'):
                result = self.analyze_file(file.raw_url, file.filename)
                results.append(result)

        return results

    def analyze_file(self, file_url: str, filename: str) -> Dict[str, Any]:
        response = requests.get(file_url)
        content = response.text

        static_results = self.run_static_analysis(content)
        ml_score = self.ml_analysis(content)

        results = {
            "filename": filename,
            **static_results,
            "ml_score": ml_score
        }

        self.redis.hmset(filename, results)
        return results

    def run_static_analysis(self, content: str) -> Dict[str, Any]:
        pylint_score = self.run_pylint(content)
        flake8_issues = self.run_flake8(content)
        security_issues = self.run_bandit(content)

        return {
            "pylint_score": pylint_score,
            "flake8_issues": flake8_issues,
            "security_issues": security_issues
        }

    @staticmethod
    def run_pylint(content: str) -> float:
        from io import StringIO
        import sys

        old_stdout = sys.stdout
        sys.stdout = StringIO()

        pylint.lint.Run(["-", "--output-format=text"], do_exit=False, stdin=content)

        output = sys.stdout.getvalue()
        sys.stdout = old_stdout

        try:
            score = float(output.split("\n")[-3].split("/")[0])
            return score
        except (IndexError, ValueError):
            return 0.0

    @staticmethod
    def run_flake8(content: str) -> List[str]:
        style_guide = flake8.get_style_guide()
        report = style_guide.input_file(
            filename=None,
            lines=content.splitlines(True)
        )
        return [f"{error}" for error in report._deferred_print]

    @staticmethod
    def run_bandit(content: str) -> List[Dict[str, Any]]:
        from bandit.core import manager

        b_mgr = manager.BanditManager(None, 'file')
        b_mgr.discover_files(['file'], recursive=False)
        b_mgr.run_tests()

        return [issue.to_dict() for issue in b_mgr.get_issue_list()]

    def ml_analysis(self, content: str) -> float:
        inputs = self.tokenizer(content, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.logits.softmax(dim=1)[0][1].item()  # Probability of positive class


def main():
    github_token = "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"
    redis_url = "redis://localhost:6379"
    reviewer = AutomatedCodeReview(github_token, redis_url)
    results = reviewer.analyze_pull_request("MarkoRrokaj/AI-powered-Code-Review-Bot", 1)
    logger.info(f"Analysis results: {results}")


if __name__ == "__main__":
    main()