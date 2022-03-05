# Benchmark script. Run with no arguments for usage info.

# Imports go here
import concurrent.futures
import json
import requests
import sys
import time

from typing import Tuple, Callable, List, Dict

import numpy as np
import pandas as pd

# Constants go here
USAGE = '''
Microbenchmark script for zero-copy model loading.  To use, first deploy the
service to test, then run this script.

Usage:
    python benchmark.py <port> <output_file>
Where:
    * <port> is the local network port on which to connect
    * <output_file> is the location where the output CSV file should go.
'''

INTENT_INPUT = {
    'context':
        ("I came here to eat chips and beat you up, "
         "and I'm all out of chips.")
}
QA_INPUT = {
    'question': 'What is 1 + 1?',
    'context':
        """Addition (usually signified by the plus symbol +) is one of the four basic 
        operations of arithmetic, the other three being subtraction, multiplication 
        and division. The addition of two whole numbers results in the total amount 
        or sum of those values combined. The example in the adjacent image shows a 
        combination of three apples and two apples, making a total of five apples. 
        This observation is equivalent to the mathematical expression "3 + 2 = 5" 
        (that is, "3 plus 2 is equal to 5").
        """
}
SENTIMENT_INPUT = {
    'context': "We're not happy unless you're not happy."
}
GENERATE_INPUT = {
    'prompt_text': 'All your base are'
}

# For now, we have a single canned input for each model type.
MODEL_INPUTS = {
    'intent': INTENT_INPUT,
    'sentiment': SENTIMENT_INPUT,
    'qa': QA_INPUT,
    'generate': GENERATE_INPUT
}

LANGUAGES = ['en', 'es', 'zh']
MODEL_TYPES = list(MODEL_INPUTS.keys())

# Map the integer model IDs from the trace to pairs of language code and
# model type.
MODEL_ID_TO_PARAMS = [
    (lang_code, model_name)
    for lang_code in LANGUAGES
    for model_name in MODEL_TYPES
]

###############################################################################
# SUBROUTINES


def call_model_rest(
        model_type: str, language: str, port_: int,
        timeout_sec: float = 5.0) \
        -> Tuple[int, float, float]:
    '''
    Callack function that calls a model deployed as a REST web service,
    retrieves the result, and returns elapsed time.

    :param model_type: Type of model to call; must be one of
                       'intent', 'sentiment', 'qa', or 'generate'
    :param language: Two-letter language code; must be one of
                     'en', 'es', 'zh'
    :param port_: Port on which the local REST API is listening.
    :param timeout_sec: Request timeout, in seconds.

    :returns: Tuple of HTTP result code and start and end times
              of the web service call. If a client-side timeout
              happens, the result code will be 408 (request timeout)
    '''
    if model_type not in MODEL_TYPES:
        raise ValueError(f'Unexpected model type "{model_type}" '
                         f'(expected {MODEL_TYPES}')
    if language not in LANGUAGES:
        raise ValueError(f'Unexpected language code "{language}" '
                         f'(expected {LANGUAGES}')

    # For now, use the same input every time
    model_input = MODEL_INPUTS[model_type]

    start_time = time.time()
    try:
        result = requests.put(
            f'http://127.0.0.1:{port_}/predictions/{model_type}_{language}',
            json.dumps(model_input),
            timeout=timeout_sec)
        end_time = time.time()
        status_code = result.status_code
    except requests.exceptions.Timeout:
        end_time = start_time + timeout_sec
        status_code = 408  # HTTP/408 Request Timeout

    return status_code, start_time, end_time


def gen_start_times(requests_per_sec: float, num_sec: int,
                    seed: int) -> np.ndarray:
    '''
    Generate a trace of inference request start times. Divides the trace
    into 1-second intervals. Each interval gets a number of requests drawn
    from a Poissson distribution. These requests are evenly spread through the
    interval.

    :param requests_per_sec: Average requests per second overall
    :param num_sec: Number of seconds of trace to generate
    :param seed: Seed for the random number generator

    :returns: Numpy array of timestamps (starting from 0) for the requests
     in the trace
    '''
    trace = []
    rng = np.random.default_rng(seed)

    # Compute the number of requests in each 1-second window.
    req_per_window = rng.poisson(requests_per_sec, size=num_sec)

    for window_num in range(num_sec):
        num_requests = req_per_window[window_num]
        if num_requests > 0:
            request_interval = 1.0 / num_requests
            for i in range(num_requests):
                trace.append(window_num + request_interval * i)

    return np.array(trace)


def gen_model_ids(lambda_: float, num_models: int, num_points: int,
                  seed: int) -> np.ndarray:
    '''
    Draw integer model IDs at random from a truncated Poisson distribution.

    :param lambda_: Primary parameter of the distribution, which also happens to
     be the mean value of the (untruncated) distribution.
    :param num_models: Number of models; generated IDs will range from 0 to
                       `num_models - 1`, inclusive.
    :param num_points: Number of random model IDs to return.
    :param seed: Seed for the random number generator

    :returns: Randomly generated model IDs for a series of requests, as a
     1D Numpy array of integers.
    '''
    rng = np.random.default_rng(seed)
    # Draw integers from a truncated Poisson distribution. Start with a
    # non-truncated distribution, then resample for
    # any values that went over the limit.
    int_ids = rng.poisson(lambda_, size=num_points)
    while np.any(int_ids >= num_models):
        new_values = rng.poisson(lambda_, size=np.sum(int_ids >= num_models))
        int_ids[int_ids >= num_models] = new_values
    return int_ids


def run_single_benchmark(
        model_callback: Callable,
        requests_per_sec: float,
        num_sec: int,
        model_id_to_params: List[Tuple[str, str]],
        model_lambda: float = 0.3,
        seed: int = 42) -> pd.DataFrame:
    '''
    A single run of the benchmark.

    Sends a stream of requests to multiple models, with the rate varying
    according to a Poisson distribution and division of traffic among models
    following a truncated Poisson distribution.

    :param model_callback: Thread-safe callback function that makes a
                           single request and returns a tuple of
                           ``(result code, start time, end time)``.
                           Should have the signature
                           `f(model_type: str, language: str)`
    :param requests_per_sec: Mean of the Poisson distribution that determines
     the number of requests in each 1-second window.
    :param num_sec: Seconds of traffic to generate at the requested rate.
     The actual session will extend past this window until all open requests
     have finished.
    :param model_lambda: Primary parameter of the truncated Poisson
     distribution used to split requests among models. Approximately
     equal to the mean of the distribution. The default value of 0.3 sends
     70% of traffic to model 0.
    :param model_id_to_params: List that maps integer model ID to a tuple of
     (language code, model name) for each of the models.
    :param seed: Seed for the random number generator

    :returns: DataFrame of benchmark results at per-request granularity
    '''
    # Preallocate the trace as a set of lists.
    benchmark_start_time = time.time()
    desired_start_times = (
        gen_start_times(requests_per_sec, num_sec, seed)
        + benchmark_start_time)
    num_requests = desired_start_times.shape[0]
    model_nums = gen_model_ids(model_lambda, len(model_id_to_params),
                               num_requests, seed)
    language_codes = [model_id_to_params[num][0] for num in model_nums]
    model_types = [model_id_to_params[num][1] for num in model_nums]
    actual_start_times = [None] * num_requests
    end_times = [None] * num_requests
    result_codes = [None] * num_requests

    # Because some notebook servers (i.e. VSCode) don't play well with
    # asyncio, we use threads to manage concurrent requests.
    thread_pool = concurrent.futures.ThreadPoolExecutor(1000)

    # Map from request object to request number
    active_requests = {}  # type: Dict[concurrent.futures.Future, int]

    # Main event loop: Spawn background requests, get their responses.
    request_num = 0
    while request_num < num_requests or len(active_requests) > 0:
        sec_to_next = (
            1.0 if request_num >= num_requests
            else desired_start_times[request_num] - time.time()
        )
        if sec_to_next <= 0:
            # Time to send the next request
            lang_code = language_codes[request_num]
            model_type = model_types[request_num]
            future = thread_pool.submit(
                model_callback, model_type, lang_code)
            active_requests[future] = request_num
            request_num += 1
        else:
            # Block until it's time to send the next request or a previous
            # request is done.
            ready_set, _ = concurrent.futures.wait(
                list(active_requests.keys()),
                timeout=sec_to_next)

            # Record timings from any open requests that have completed.
            for future in ready_set:
                request_id = active_requests.pop(future)
                result_code, start_time, end_time = future.result()
                actual_start_times[request_id] = start_time
                end_times[request_id] = end_time
                result_codes[request_id] = result_code

    # Collate results as a DataFrame
    result = pd.DataFrame({
        'request_id': range(num_requests),
        'model_num': model_nums,
        'lang_code': language_codes,
        'model_type': model_types,
        'desired_start': desired_start_times,
        'actual_start': actual_start_times,
        'end': end_times,
        'result_code': result_codes
    })

    # Make all times relative to start of the trace
    for key in ("desired_start", "actual_start", "end"):
        result[key] -= benchmark_start_time
    result["latency"] = result["end"] - result["actual_start"]

    return result


def run_benchmarks(
        model_callback: Callable,
        num_sec: int = 60,
        min_request_rate: int = 2,
        request_rate_step: float = 0.5,
        max_failure_fraction: float = 0.6) -> pd.DataFrame:
    '''
    Perform multiple runs of the benchmark, increasing the request
    rate gradually until requests start returning errors.

    :param model_callback: Thread-safe callback function that makes a
                           single request and returns a tuple of
                           ``(result code, start time, end time)``.
                           Should have the signature
                           `f(model_type: str, language: str)`
    :param num_sec: Seconds of traffic to generate for each run.
                    The actual session will extend past this window
                    until all open requests have finished.
    :param min_request_rate: Mean request rate for the first run of the
                             benchmark.
                             The actual request rate will follow a Poisson
                             distribution with this mean.
    :param request_rate_step: Amount by which the request rate increases
                              with each subsequent run of the benchmark,
                              in requests per second.
    :param max_failure_fraction: What fraction of failed web service calls
                                 the benchmark will tolerate per run before
                                 stopping the overall process.

    :returns: A Pandas DataFrame of detailed timings for all web service
              requests. The column ``request_rate`` tells which run of the
              benchmark each request belongs to.
    '''
    to_concat = []
    request_rate = min_request_rate
    failure_fraction = 0.

    while failure_fraction <= max_failure_fraction:
        print(f'Running at {request_rate} requests/sec.')
        times = run_single_benchmark(model_callback,
                                     request_rate, num_sec,
                                     MODEL_ID_TO_PARAMS)
        times.insert(0, 'request_rate', request_rate)
        to_concat.append(times)
        num_failures = sum(times['result_code'] != 200)
        num_requests = len(times.index)
        failure_fraction = num_failures / num_requests
        print(f' => {failure_fraction * 100.:0.1f}% failure rate')
        request_rate += request_rate_step

    print(f'Stopping due to fraction of failures ({failure_fraction}) '
          f'exceeding allowable limit ({max_failure_fraction})')
    return pd.concat(to_concat)


###############################################################################
# MAIN
if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(USAGE)
        exit(-1)
    port, output_file = sys.argv[1:]
    print(f'port is {port} and output CSV file is {output_file}')

    # For now, every model is a web service call and the only thing that
    # changes is the port.
    def callback_fn(model_type: str, language: str):
        return call_model_rest(model_type, language, port)

    results = run_benchmarks(callback_fn)

    results.to_csv(output_file)






