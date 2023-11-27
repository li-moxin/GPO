import random, logging, time
from faulthandler import disable
from automatic_prompt_engineer import data, llm


def get_query(prompt_gen_template, demos_template, subsampled_data):
    """
    Returns a query for the prompt generator. A query is the prompt that is sent to the LLM.
    Parameters:
        prompt_gen_template: The template for the prompt generator queries.
        demos_template: The template for the demonstrations.
        subsampled_data: The data to use for the demonstrations.
    Returns:
        A query for the prompt generator.
    """
    inputs, outputs = subsampled_data
    demos = demos_template.fill(subsampled_data)
    return prompt_gen_template.fill(input=inputs[0], output=outputs[0], full_demo=demos)

def get_query_naive(prompt_gen_template, demos_template_for_gen, subsampled_data_1, subsampled_data_2):
    """
    Returns a query for the prompt generator. A query is the prompt that is sent to the LLM.
    Parameters:
        prompt_gen_template: The template for the prompt generator queries.
        demos_template: The template for the demonstrations.
        subsampled_data: The data to use for the demonstrations.
    Returns:
        A query for the prompt generator.
    """
    inputs, outputs = subsampled_data_1
    demos= demos_template_for_gen.fill(source=subsampled_data_1, target=subsampled_data_2)
    return prompt_gen_template.fill(input=inputs[0], output=outputs[0], full_demo=demos)


def paired_shuffle(data):
    data = list(zip(data[0], data[1]))
    random.shuffle(data)
    data = list(zip(*data))
    return data
    

def generate_prompts(prompt_gen_template, demos_template, prompt_gen_data, config):
    """
    Generates prompts using the prompt generator.
    Parameters:
        prompt_gen_template: The template for the prompt generator queries.
        demos_template: The template for the demonstrations.
        prompt_gen_data: The data to use for prompt generation.
        config: The configuration dictionary.
    Returns:
        A list of prompts.
    """
    queries = []
    num_train_data = config['num_subsamples'] * config['num_demos']
    all_subsampled_data = data.subsample_data(prompt_gen_data, num_train_data, config) # this step does not sample.
    
    replicate_time = 1
    if 'replicate_time' in config:
        replicate_time = config['replicate_time'] # not used here
        
    # logging.info('replicating {} times'.format(replicate_time))

    for _ in range(replicate_time):
        if replicate_time > 1:
            all_subsampled_data = paired_shuffle(all_subsampled_data)
        for i in range(config['num_subsamples']):
            subsampled_data = all_subsampled_data[0][i * config['num_demos'] : (i+1) * config['num_demos']], all_subsampled_data[1][i * config['num_demos'] : (i+1) * config['num_demos']]
            queries.append(get_query(prompt_gen_template, demos_template, subsampled_data))
    # Instantiate the LLM
    model = llm.model_from_config(config['model'], disable_tqdm=False)
    prompts = model.generate_text(
        queries, n=config['num_prompts_per_subsample'])
    return prompts

def generate_prompts_naive(prompt_gen_template, demos_template_for_gen, prompt_gen_data, config):
    """
    Generates prompts using the prompt generator.
    Parameters:
        prompt_gen_template: The template for the prompt generator queries.
        demos_template: The template for the demonstrations.
        prompt_gen_data: The data to use for prompt generation.
        config: The configuration dictionary.
    Returns:
        A list of prompts.
    """
    prompt_gen_data_1, prompt_gen_data_2 = prompt_gen_data[0], prompt_gen_data[1]
    queries = []
    num_train_data = config['num_subsamples'] * config['num_demos']
    
    for i in range(config['num_subsamples']):
        subsampled_data_1 = prompt_gen_data_1[0][i * config['num_demos'] : (i+1) * config['num_demos']], prompt_gen_data_1[1][i * config['num_demos'] : (i+1) * config['num_demos']]
        subsampled_data_2 = prompt_gen_data_2[0][i * config['num_demos'] : (i+1) * config['num_demos']], prompt_gen_data_2[1][i * config['num_demos'] : (i+1) * config['num_demos']]
        queries.append(get_query_naive(prompt_gen_template, demos_template_for_gen, subsampled_data_1, subsampled_data_2))

    # Instantiate the LLM
    model = llm.model_from_config(config['model'], disable_tqdm=False)
    prompts = model.generate_text(
        queries, n=config['num_prompts_per_subsample'])
    return prompts

