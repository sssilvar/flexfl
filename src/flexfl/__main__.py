import os
import sys
import yaml
import logging
import click

import flexfl
from flexfl.app import App

if sys.stdout.isatty():
# You're running in a real terminal
    LOG_FORMAT="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
else:
    LOG_FORMAT="%(name)s - %(levelname)s - %(message)s"

logging.basicConfig(
    level=getattr(logging, os.getenv("LOGLEVEL", "INFO").upper()),
    format=LOG_FORMAT,
    datefmt="%Y-%m-%dT%H:%M:%S"
)

logger = logging.getLogger("flexfl")

# set levels for other modules
logging.getLogger("urllib3").setLevel(logging.WARNING)

@click.group()
@click.version_option(package_name="FlexFL")
def cli():
    pass

@cli.command()
@click.option("-v", "--verbose", count=True)
def version(verbose):
    """Displays the version"""
    click.echo("Version: %s" % flexfl.__version__)
    if verbose > 0:
        click.echo("Author: %s" % flexfl.__author__)


@cli.command()
@click.option(
    "-c",
    "--conf",
    "--config",
    "config_file",
    type=click.Path(exists=True),
)
def serve(config_file):
    """Start FlexFL in server mode"""
    
    settings = {}
    if config_file:
        try:
            with open(config_file, "r") as stream:
                try:
                    settings = yaml.safe_load(stream)
                except yaml.YAMLError as exc:
                    click.echo(exc)
                    sys.exit(1)
        except IOError as exc:
            logger.fatal("%s: %s", exc.strerror, exc.filename)
            sys.exit(1)
        except Exception as exc:
            logger.fatal(
                "Cannot load conf file '%s'. Error message is: %s", config_file, exc
            )
            sys.exit(1)

    # TODO: Create your application object
    app = App(settings)
    try:
        logger.info("Starting")
        # TODO: Start your application
        app.start()
    except KeyboardInterrupt:
        pass
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("Unexpected exception: %s", exc)
    finally:
        logger.info("Shutting down")
        # TODO: Cleanup code
        app.stop()

    logger.info("All done")


if __name__ == "__main__":
    cli(prog_name="FlexFL")
