# import tqdm
from args import get_args
from prompt_strategies.Zeroshot import Zeroshot_generate_result
from prompt_strategies.Fewshot import Fewshot_generate_result
from prompt_strategies.Zeroshot_CoT import Zeroshot_CoT_generate_result
from prompt_strategies.Fewshot_CoT import Fewshot_CoT_generate_result
from prompt_strategies.SelfDebug import SelfDebug_generate_result
from prompt_strategies.Reflection import Reflection_generate_result
from prompt_strategies.SelfPlan import SelfPlan_generate_result
from prompt_strategies.ProgressiveHint import ProgressiveHint_generate_result
from prompt_strategies.Persona import Persona_generate_result

def main(args):
    if args.strategy == 'Zeroshot':
        Zeroshot_generate_result(args)
    elif args.strategy == 'Fewshot':
        Fewshot_generate_result(args)
    elif args.strategy == 'Zeroshot_CoT':
        Zeroshot_CoT_generate_result(args)
    elif args.strategy == 'Fewshot_CoT':
        Fewshot_CoT_generate_result(args)
    elif args.strategy == 'SelfDebug':
        SelfDebug_generate_result(args)
    elif args.strategy == 'Reflection':
        Reflection_generate_result(args)
    elif args.strategy == 'SelfPlan':
        SelfPlan_generate_result(args)
    elif args.strategy == 'ProgressiveHint':
        ProgressiveHint_generate_result(args)
    elif args.strategy == 'Persona':
        Persona_generate_result(args)


    

if __name__ == "__main__":
    args = get_args()
    main(args)