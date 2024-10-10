import importlib

class ModelLoader:
    def __init__(self, model_name, config, use_accel):
        self.model_name = model_name
        self.config = config
        self._model = None
        self.use_accel = use_accel

    def _lazy_import(self, module_name, func_name):
        """Dynamically import a module and return the desired function."""
        if module_name.startswith('.'):
            # Convert relative import to absolute import based on the current package context
            module_name = __package__ + module_name
        module = importlib.import_module(module_name)
        return getattr(module, func_name)

    @property
    def model(self):
        """Load and return the model instance, if not already loaded."""
        if self._model is None:
            load_func = self._lazy_import(self.config['load'][0], self.config['load'][1])
            if self.config.get('call_type') == 'api':
                self._model = load_func(
                    self.config['model_path_or_name'], 
                    self.config['base_url'], 
                    self.config['api_key'], 
                    self.config['model']
                )
            else:
                self._model = load_func(self.model_name, self.config, use_accel=self.use_accel)
        return self._model

    @property
    def infer(self):
        """Return the inference function."""
        return self._lazy_import(self.config['infer'][0], self.config['infer'][1])

class ModelRegistry:
    def __init__(self):
        self.models = {}

    def register_model(self, name, config, use_accel):
        """Register a model configuration."""
        self.models[name] = ModelLoader(name, config, use_accel)

    def load_model(self, choice, use_accel=False):
        """Load a model based on the choice."""
        if choice in self.models:
            return self.models[choice].model
        else:
            raise ValueError(f"Model choice '{choice}' is not supported.")

    def infer(self, choice):
        """Get the inference function for a given model."""
        if choice in self.models:
            return self.models[choice].infer
        else:
            raise ValueError(f"Inference choice '{choice}' is not supported.")

# Initialize model registry
model_registry = ModelRegistry()

# Configuration of models
model_configs = {
    'gpt4o': {
        'load': ('.api', 'load_model'),
        'infer': ('.api', 'infer'),
        'model_path_or_name': 'GPT4o',
        'base_url': None,
        'api_key': None,
        'model': 'gpt-4o',
        'call_type': 'api'
    },
    'claude-3-5-sonnet-20240620': {
        'load': ('.api', 'load_model'),
        'infer': ('.api', 'infer'),
        'model_path_or_name': 'claude-3-5-Sonnet-20240620',
        'base_url': None,
        'api_key': None,
        'model': 'claude-3-5-sonnet-20240620',
        'call_type': 'api'
    },
    'gemini-1.5-pro': {
        'load': ('.api', 'load_model'),
        'infer': ('.api', 'infer'),
        'model_path_or_name': 'Gemini-1.5-Pro',
        'base_url': None,
        'api_key': None,
        'model': 'gemini-1.5-pro',
        'call_type': 'api'
    },
    'glm-4-v-plus': {
        'load': ('.glm_v_api', 'load_model'),
        'infer': ('.glm_v_api', 'infer'),
        'model_path_or_name': 'glm-4v-plus',
        'base_url': None,
        'api_key': None,
        'model': 'glm-4v-plus',
        'call_type': 'api'
    },
    'qwen-vl-max': {
        'load': ('.api', 'load_model'),
        'infer': ('.api', 'infer'),
        'model_path_or_name': 'qwen-vl-max',
        'base_url': None,
        'api_key': None,
        'model': 'qwen-vl-max',
        'call_type': 'api'
    },
    'cogvlm2-llama3-chinese-chat-19B': {
        'load': ('.cogvlm2','load_model'),
        'infer': ('.cogvlm2','infer'),
        'model_path_or_name': '/path/to/cogvlm2-llama3-chat-19B',
        'call_type': 'local',
        'tp': 8
    },
    'MiniCPM-Llama3-V-2.5': {
        'load': ('.lmdeploy_chat', 'load_model'),
        'infer': ('.lmdeploy_chat', 'infer'),
        'model_path_or_name': '/path/to/MiniCPM-Llama3-V-2_5',
        'call_type': 'local',
        'tp': 1
    },
    'MiniCPM-V-2.6': {
        'load': ('.api', 'load_model'),
        'infer': ('.api', 'infer'),
        'model_path_or_name': '/path/to/MiniCPM-V-2_6',
        'call_type': 'local',
        'tp': 1
    },
    'Qwen2-VL-7B': {
        'load': ('.api', 'load_model'),
        'infer': ('.api', 'infer'),
        'base_url': None,
        'api_key': None,
        'model': None,
        'call_type': 'api'
    },
    'Qwen2-VL-72B': {
        'load': ('.api', 'load_model'),
        'infer': ('.api', 'infer'),
        'base_url': None,
        'api_key': None,
        'model': None,
        'call_type': 'api'
    },
    'glm-4v-9b': {
        'load': ('.glm_4v', 'load_model'),
        'infer': ('.glm_4v', 'infer'),
        'model_path_or_name': '/path/to/glm-4v-9b',
        'call_type': 'local',
        'tp': 8
    },
    'idefics2-8b': {
        'load': ('.idefics2', 'load_model'),
        'infer': ('.idefics2', 'infer'),
        'model_path_or_name': '/path/to/idefics2-8b',
        'call_type': 'local',
        'tp': 1
    },
    'llava-v1.6-34b': {
        'load': ('.api', 'load_model'),
        'infer': ('.api', 'infer'),
        'base_url': None,
        'api_key': None,
        'model': None,
        'call_type': 'api'
    },
    'llava-next-72b-hf': {
        'load': ('.api', 'load_model'),
        'infer': ('.api', 'infer'),
        'base_url': None,
        'api_key': None,
        'model': None,
        'call_type': 'api'
    },
    'InternVL2-8B': {
        'load': ('.lmdeploy_chat', 'load_model'),
        'infer': ('.lmdeploy_chat', 'infer'),
        'model_path_or_name': '/path/to/InternVL2-8B',
        'call_type': 'local',
        'tp': 1
    },
    'InternVL2-40B': {
        'load': ('.lmdeploy_chat', 'load_model'),
        'infer': ('.lmdeploy_chat', 'infer'),
        'model_path_or_name': '/path/to/InternVL2-40B',
        'call_type': 'local',
        'tp': 4
    },
    'InternVL2-Llama3-76B': {
        'load': ('.lmdeploy_chat', 'load_model'),
        'infer': ('.lmdeploy_chat', 'infer'),
        'model_path_or_name': '/path/to/InternVL2-Llama3-76B',
        'call_type': 'local',
        'tp': 4
    },
    'Qwen-VL-Chat': {
        'load': ('.qwen_vl_chat', 'load_model'),
        'infer': ('.qwen_vl_chat', 'infer'),
        'model_path_or_name': '/path/to/Qwen-VL-Chat',
        'call_type': 'local',
        'tp': 8
    },
}


def load_model(choice, use_accel=False):
    """Load a specific model based on the choice."""
    model_registry.register_model(choice, model_configs[choice], use_accel)
    return model_registry.load_model(choice, use_accel)

def infer(choice):
    """Get the inference function for a specific model."""
    return model_registry.infer(choice)

