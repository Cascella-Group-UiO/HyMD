from hPF.input_parser import Config, read_config_toml, parse_config_toml


def test_input_parser_read_config_toml(config_toml):
    config_toml_file, config_toml_str = config_toml
    file_content = read_config_toml(config_toml_file)
    assert config_toml_str == file_content


def test_input_parser_file(config_toml):
    _, config_toml_str = config_toml
    config = parse_config_toml(config_toml_str)
    assert isinstance(config, Config)
