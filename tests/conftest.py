def pytest_addoption(parser):
    parser.addoption("--data_root", action="store", default=".")


def pytest_generate_tests(metafunc):
    def register(option):
        option_value = metafunc.config.getoption(option)
        if option in metafunc.fixturenames and option_value is not None:
            metafunc.parametrize(option, [option_value])

    register("data_root")
