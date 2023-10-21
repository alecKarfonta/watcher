import logggin

class Watcher():


    def __init__(
            self,
            logger:logging.Logger=None, 
            config_filename:str="qa_config.json", 
            pre_init:bool=False
        ):
        # If provided a logger
        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger()
        # Log init
        self.logger.debug(f"{__class__.__name__}.init()")

        # Load config
        try:
            self.config = json.load(open(config_filename, "r"))
        except FileNotFoundError:
            self.logger.error(f"{__class__.__name__}.init(): FileNotFoundError Could not read config file at {config_filename}")
            return None
        
        self.model = None
        self.sent_sim = None
        # If should pre initialize model
        if pre_init:
            self.init_model()