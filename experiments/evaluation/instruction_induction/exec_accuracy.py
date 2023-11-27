import numpy as np
from collections import Counter
import random, logging, time, copy

from automatic_prompt_engineer import data, llm, evaluate, template
from experiments.evaluation.instruction_induction import utility
from experiments.evaluation.instruction_induction.utility import normalize_prediction


def get_query(prompt, eval_template, input_, output_, demo_data, demos_template):
    demos = demos_template.fill(demo_data)
    query = eval_template.fill(prompt=prompt,
                               input=input_,
                               output='',
                               full_demo=demos)
    return query


def get_batched_query(prompt, eval_template, eval_data, demo_data, demos_template):
    demos = demos_template.fill(demo_data)
    query = ''
    for i, data in enumerate(eval_data):
        input_, output_ = data[0], data[1]
        query += 'No. ' + str(i+1) + '\n\n' + eval_template.fill(prompt=prompt, input=input_, output='', full_demo=demos) + '\n\n'
    return query

def voting(aggregated_outputs, answers):
    num_prompts = len(aggregated_outputs)
    num_test_data = len(aggregated_outputs[0])
    new_outputs = []
    confidences =[]
    results = []
    for i in range(num_test_data):
        normalized_preds = [normalize_prediction(aggregated_outputs[j][i]) for j in range(num_prompts)]
        cnt = Counter(normalized_preds)
        if type(answers[i]) == list:
            normalized_answers = [normalize_prediction(a) for a in answers[i]]
        else:
            normalized_answers = [normalize_prediction(answers[i])]
        logging.info("{} {}".format(str(cnt), str(normalized_answers)))

        max_val = max(cnt.values())
        target_outputs = []
        
        for k, v in cnt.items():
            if v == max_val:
                target_outputs.append(k)

        new_outputs.append(target_outputs[0])
        confidence_value = cnt[target_outputs[0]] / num_prompts
        confidences.append(confidence_value)
        results.append([normalized_preds, normalized_answers, target_outputs[0], confidence_value])
    return new_outputs, confidences, results

def label_data(prompts, eval_template, eval_data, demos_template, few_shot_data, config):
    eval_template = template.EvalTemplate(eval_template)
    demos_template = template.DemosTemplate(demos_template)
    
    queries = []
    answers = []
    data_num = len(eval_data[0])
    logging.info("Labeling {} data".format(data_num))
    logging.info("Labeling with these prompts")
    logging.info(prompts)
    for prompt in prompts:
        for d in zip(*eval_data):
            input_, output_ = d
            demo_data = [[], []]
            query = get_query(
                prompt, eval_template, input_, output_, demo_data, demos_template)
            queries.append(query)
            answers.append(output_)
    
    model = llm.model_from_config(config['model'])
    model_outputs = model.generate_text(queries, 1)
    
    task = config['task']
    metric = utility.TASK_TO_METRIC.get(task, utility.default_metric)

    logging.info(f'Using metric "{metric}" for task "{task}"...')
    
    if metric == 'f1':
        score_fn = utility.get_multi_answer_f1
    elif metric == 'es':
        score_fn = utility.get_multi_answer_exact_set
    elif metric == 'contains':
        score_fn = utility.get_multi_answer_contains
    elif metric == 'em':
        score_fn = utility.get_multi_answer_em

    # prompt * data_num
    model_outputs = [model_outputs[i* data_num:(i+1) * data_num] for i in range(len(prompts))]
    
    if 'ensemble' in config and config['ensemble']:
        logging.info("Ensembling prompts")
        model_outputs, confidences, _ = voting(model_outputs, answers[:data_num])
        ensemble_acc =[]
        for i in range(data_num):
            if type(answers[i]) == list:
                normalized_ans = normalize_prediction(answers[i][0])
            else:
                normalized_ans = normalize_prediction(answers[i])
            ensemble_acc.append(score_fn(model_outputs[i], [normalized_ans]))
        logging.info("Ensemble label acc {}".format(np.mean(ensemble_acc)))
        conf_lower_bound = -1
        if 'conf_lower_bound' in config and config['conf_lower_bound']:
            logging.info("Selecting at confidence lower bound {}".format(config['conf_lower_bound']))

            conf_lower_bound = config['conf_lower_bound']
        if type(eval_data[1][0]) == list:
            model_outputs = [[line] for line in model_outputs]
        model_outputs = [[eval_data[0][i] ,model_outputs[i]] for i in range(data_num) if confidences[i] > conf_lower_bound]
        ensemble_acc = [ensemble_acc[i] for i in range(data_num) if confidences[i] > conf_lower_bound]
        logging.info("Labeled data # {} total {} acc {} ".format(len(model_outputs), len(eval_data[0]), np.mean(ensemble_acc)))
        # upsampling
        if len(model_outputs) < len(eval_data[0]):
            rep_times = data_num // len(model_outputs)
            fill_up_model_outputs_num = data_num - len(model_outputs) * rep_times
            model_outputs = model_outputs * rep_times + random.sample(model_outputs, fill_up_model_outputs_num)
            random.shuffle(model_outputs)
        model_outputs = [line[0] for line in model_outputs], [line[1] for line in model_outputs]
        assert len(model_outputs) == 2
        assert len(model_outputs[1]) == data_num
    else:
        logging.info("Not emsembling, using top 1 prompt")
        model_outputs = model_outputs[0]
        answers = answers[:data_num]
        normalized_model_outputs = [normalize_prediction(s) for s in model_outputs]
        normalized_answers = [normalize_prediction(s[0]) if type(s) == list else normalize_prediction(s) for s in answers]
        logging.info("Top 1 label acc {}".format(sum( [1 if normalized_answers[i] == normalized_model_outputs[i] else 0 for i in range(data_num)] ) / data_num ))
        model_outputs = [[eval_data[0][i] ,model_outputs[i]] for i in range(data_num)]
        model_outputs = [line[0] for line in model_outputs], [line[1] for line in model_outputs]
    return model_outputs


def label_data_vicuna(prompts, eval_template, eval_data, demos_template, few_shot_data, config):
    eval_template = template.EvalTemplate(eval_template)
    demos_template = template.DemosTemplate(demos_template)
    
    queries = []
    answers = []
    data_num = len(eval_data[0])
    logging.info("Labeling {} data".format(data_num))
    logging.info("Labeling with these prompts")
    logging.info(prompts)
    # Preparing queries
    for prompt in prompts:
        for d in zip(*eval_data):
            input_, output_ = d
            demo_data = [[], []]
            query = get_query(
                prompt, eval_template, input_, output_, demo_data, demos_template)
            queries.append(query)
            answers.append(output_)
    
    # LLM generationg
    model = llm.model_from_config(config['model'])
    model_outputs = model.generate_text(queries, 1)

    task = config['task']
    metric = utility.TASK_TO_METRIC.get(task, utility.default_metric)

    def model_output_to_label(s):
        s_n = normalize_prediction(s)
        for label in ['positive', 'negative', 'neutral']:
            if label in s_n:
                return label
        return s
    
    # mapping outputs to labels. for vicuna.
    model_outputs = [model_output_to_label(l) for l in model_outputs]

    logging.info(f'Using metric "{metric}" for task "{task}"...')
    
    if metric == 'f1':
        score_fn = utility.get_multi_answer_f1
    elif metric == 'es':
        score_fn = utility.get_multi_answer_exact_set
    elif metric == 'contains':
        score_fn = utility.get_multi_answer_contains
    elif metric == 'em':
        score_fn = utility.get_multi_answer_em
        
    # prompt * data_num
    model_outputs = [model_outputs[i* data_num:(i+1) * data_num] for i in range(len(prompts))]
    
    if 'ensemble' in config and config['ensemble']:
        logging.info("Ensembling prompts")
        model_outputs, confidences, _ = voting(model_outputs, answers[:data_num])
        ensemble_acc =[]
        for i in range(data_num):
            if type(answers[i]) == list:
                normalized_ans = normalize_prediction(answers[i][0])
            else:
                normalized_ans = normalize_prediction(answers[i])
            ensemble_acc.append(score_fn(model_outputs[i], [normalized_ans]))
        logging.info("Ensemble label acc {}".format(np.mean(ensemble_acc)))

        conf_lower_bound = -1
        if 'conf_lower_bound' in config and config['conf_lower_bound']:
            logging.info("Selecting at confidence lower bound {}".format(config['conf_lower_bound']))

            conf_lower_bound = config['conf_lower_bound']
        if type(eval_data[1][0]) == list:
            model_outputs = [[line] for  line in model_outputs]
        model_outputs = [[eval_data[0][i] ,model_outputs[i]] for i in range(data_num) if confidences[i] > conf_lower_bound]
        ensemble_acc = [ensemble_acc[i] for i in range(data_num) if confidences[i] > conf_lower_bound]
        logging.info("Labeled data num {} {} {} ".format(len(model_outputs), len(eval_data[0]), np.mean(ensemble_acc)))

        if len(model_outputs) < len(eval_data[0]):
            rep_times = data_num // len(model_outputs)
            fill_up_model_outputs_num = data_num - len(model_outputs) * rep_times
            model_outputs = model_outputs * rep_times + random.sample(model_outputs, fill_up_model_outputs_num)
            random.shuffle(model_outputs)
        model_outputs = [line[0] for line in model_outputs], [line[1] for line in model_outputs]
        assert len(model_outputs) == 2
        assert len(model_outputs[1]) == data_num
    else:
        logging.info("Not emsembling, using top 1 prompt")

        model_outputs = model_outputs[0]
        answers = answers[:data_num]
        normalized_model_outputs = [normalize_prediction(s) for s in model_outputs]
        normalized_answers = [normalize_prediction(s[0]) if type(s) == list else normalize_prediction(s) for s in answers]
        logging.info("Top 1 label acc {}".format(sum( [1 if normalized_answers[i] == normalized_model_outputs[i] else 0 for i in range(data_num)] ) / data_num ))

        model_outputs = [[eval_data[0][i] ,model_outputs[i]] for i in range(data_num)]
        model_outputs = [line[0] for line in model_outputs], [line[1] for line in model_outputs]
    return model_outputs


def exec_accuracy_evaluator(prompts, eval_template, eval_data, demos_template, few_shot_data, config):
    queries = []
    answers = []
    # no subsample now.
    subsampled_data = data.subsample_data(eval_data, config['num_samples'], config)
     
    logging.info(prompts)

    # preparing queries
    for prompt in prompts:
        for d in zip(*subsampled_data):
            input_, output_ = d
            #demo_data = data.subsample_fewshot_data(
            #    few_shot_data, config['num_few_shot'])
            demo_data = [[], []]
            query = get_query(
                prompt, eval_template, input_, output_, demo_data, demos_template)
            queries.append(query)
            answers.append(output_)
           
    # Instantiate the LLM
    model = llm.model_from_config(config['model'])
    model_outputs = model.generate_text(queries, 1)

    task = config['task']
    metric = utility.TASK_TO_METRIC.get(task, utility.default_metric)
    
    def model_output_to_label(s):
        s_n = normalize_prediction(s)
        for label in ['positive', 'negative', 'neutral']:
            if label in s_n:
                return label
        return s
    
    # only for vicuna
    if 'force_contains' in config and config['force_contains']:

        logging.info("Transforming model output to label")
        model_outputs = [model_output_to_label(l) for l in model_outputs]
        
    logging.info(f'Using metric "{metric}" for task "{task}"...')

    
    if metric == 'f1':
        score_fn = utility.get_multi_answer_f1
    elif metric == 'es':
        score_fn = utility.get_multi_answer_exact_set
    elif metric == 'contains':
        score_fn = utility.get_multi_answer_contains
    elif metric == 'em':
        score_fn = utility.get_multi_answer_em

    scores = []

    for prediction, ans_ in zip(model_outputs, answers):
        score = score_fn(prediction, ans_)
        scores.append(score)
    
    if 'ensemble' in config and config['ensemble']:
        model_outputs = [model_outputs[i*config['num_samples']: (i+1)*config['num_samples']] for i in range(len(prompts))]
        answers = answers[:config['num_samples']]
        model_outputs, confidences, results = voting(model_outputs, answers)
        prompts.append("Ensemble")
        
        conf_index = 0
        conf_high_acc = []
 
        for prediction, ans_ in zip(model_outputs, answers):
            score = score_fn(prediction, ans_)
            scores.append(score)
            
    # num_prompts x num_samples
    scores = np.array(scores).reshape(len(prompts), -1)

    res = ExecAccuracyEvaluationResult(prompts, scores)
    return res
    
class ExecAccuracyEvaluationResult(evaluate.EvaluationResult):

    def __init__(self, prompts, scores):
        self.prompts = prompts
        self.scores = scores

    def _agg_scores(self, method):
        """For each prompt, compute a statistic of the scores (e.g., mean, median)"""
        if method == 'mean':
            return [np.mean(s) for s in self.scores]
        elif method == 'median':
            return [np.median(s) for s in self.scores]
        elif method == 'std':
            return [np.std(s) for s in self.scores]
        elif method == 'max':
            return [np.max(s) for s in self.scores]
        elif method == 'min':
            return [np.min(s) for s in self.scores]
        elif method == 'iqm':
            return [np.mean(np.percentile(lps, [25, 75])) for lps in self.scores]
        else:
            raise ValueError('Invalid method: {}'.format(method))

    def sorted(self, method='default'):
        if method == 'default':
            scores = self._agg_scores('mean')
        else:
            scores = self._agg_scores(method)
        # Sort prompts by score
        sorted_prompts = [p for _, p in sorted(zip(scores, self.prompts))]
        sorted_scores = sorted(scores)
        # Reverse both and convert to lists
        sorted_prompts = list(reversed(sorted_prompts))
        sorted_scores = list(reversed(sorted_scores))
        return sorted_prompts, sorted_scores

    def in_place(self, method='default'):
        if method == 'default':
            scores = self._agg_scores('mean')
        else:
            scores = self._agg_scores(method)
        return self.prompts, scores
    
