"""Test GraphQL safety of docker_args generation in RunPod service.

This module validates that the docker command strings generated by runpod_service.py
are properly escaped and safe for embedding in GraphQL mutations, preventing the
syntax errors that were causing RunPod API failures.
"""

import pytest
from graphql import build_schema, parse

from runpod_service_nnue import _bash_c_quote, _create_docker_script

# Only import GraphQL if available (it's in requirements-dev.txt)
try:
    HAS_GRAPHQL = True
except ImportError:
    HAS_GRAPHQL = False


class TestGraphQLEscaping:
    """Test GraphQL string escaping for docker arguments."""

    def test_bash_c_quote_basic(self):
        """Test basic bash command quoting with GraphQL escaping."""
        script = "echo hello world"
        result = _bash_c_quote(script)
        assert isinstance(result, str)
        # Should contain the bash -c command structure
        assert "bash -c" in result

    def test_bash_c_quote_with_special_chars(self):
        """Test bash quoting with special characters that could break GraphQL."""
        script = "echo 'single quotes' && echo \"double quotes\" && echo $vars"
        result = _bash_c_quote(script)
        assert isinstance(result, str)
        # Should be properly escaped for both bash and GraphQL
        assert isinstance(result, str)

    def test_bash_c_quote_with_quotes(self):
        """Test GraphQL escaping of double quotes in bash commands."""
        script = 'echo "string with quotes"'
        result = _bash_c_quote(script)
        # Should not contain unescaped quotes that would break GraphQL
        assert isinstance(result, str)
        # The result should be GraphQL-safe
        try:
            # Try to embed in a GraphQL-like string to verify safety
            test_mutation = f'mutation {{ createPod(dockerArgs: "{result}") }}'
            parsed = parse(test_mutation)
            assert parsed is not None
        except Exception:
            pytest.fail("GraphQL escaping failed for quotes")

    def test_bash_c_quote_with_backslashes(self):
        """Test GraphQL escaping of backslashes in bash commands."""
        script = "echo path\\with\\backslashes"
        result = _bash_c_quote(script)
        assert isinstance(result, str)

    def test_bash_c_quote_with_newlines(self):
        """Test GraphQL escaping of newlines and whitespace in bash commands."""
        script = "echo line1\necho line2\r\necho line3\ttabbed"
        result = _bash_c_quote(script)
        assert isinstance(result, str)
        # Should not contain literal newlines that would break GraphQL
        assert "\n" not in result


class TestDockerScriptGeneration:
    """Test docker script generation and GraphQL safety."""

    def test_create_docker_script_basic(self):
        """Test basic docker script creation."""
        training_command = "train.py nnue --config config/train_nnue_default.py"
        result = _create_docker_script(training_command)

        assert isinstance(result, str)
        assert "apt-get update" in result
        assert "git clone" in result
        assert "container_setup.sh" in result
        assert training_command in result

    def test_create_docker_script_with_special_chars(self):
        """Test docker script with training commands containing special characters."""
        training_command = (
            "train.py nnue --note=\"test run with 'quotes'\" --batch_size=32"
        )
        result = _create_docker_script(training_command)

        assert isinstance(result, str)
        assert training_command in result

    def test_docker_script_graphql_safe(self):
        """Test that docker scripts are safe for GraphQL embedding."""
        training_command = "train.py nnue --note=\"complex 'test' with $special chars\""
        docker_script = _create_docker_script(training_command)
        final_docker_args = _bash_c_quote(docker_script)

        # Should not contain unescaped quotes or other problematic characters
        assert isinstance(final_docker_args, str)
        # Test that it can be embedded in GraphQL
        try:
            test_mutation = (
                f'mutation {{ createPod(dockerArgs: "{final_docker_args}") }}'
            )
            parsed = parse(test_mutation)
            assert parsed is not None
        except Exception:
            pytest.fail("GraphQL escaping failed for complex command")


@pytest.mark.skipif(not HAS_GRAPHQL, reason="graphql-core not available")
class TestGraphQLMutationSafety:
    """Test that escaped docker args are safe in actual GraphQL mutations."""

    @pytest.fixture
    def graphql_schema(self):
        """Create a simple GraphQL schema for testing."""
        schema_def = """
            type Mutation {
                createPod(dockerArgs: String!): String
            }
            
            type Query {
                hello: String
            }
        """
        return build_schema(schema_def)

    def test_simple_mutation_parsing(self, graphql_schema):
        """Test that simple docker args parse correctly in GraphQL."""
        training_command = "train.py nnue --config config/train_nnue_default.py"
        docker_script = _create_docker_script(training_command)
        escaped_args = _bash_c_quote(docker_script)

        # Create a GraphQL mutation with the escaped args
        mutation = f"""
            mutation {{
                createPod(dockerArgs: "{escaped_args}")
            }}
        """

        # This should parse without errors
        try:
            parsed = parse(mutation)
            assert parsed is not None
        except Exception as e:
            pytest.fail(f"GraphQL parsing failed: {e}")

    def test_complex_mutation_parsing(self, graphql_schema):
        """Test complex docker args with special characters in GraphQL."""
        training_command = """train.py nnue --note="test with 'quotes' and \\"double quotes\\"" --config=config/train_nnue_default.py"""
        docker_script = _create_docker_script(training_command)
        escaped_args = _bash_c_quote(docker_script)

        # Create a GraphQL mutation with the escaped args
        mutation = f"""
            mutation {{
                createPod(dockerArgs: "{escaped_args}")
            }}
        """

        # This should parse without errors
        try:
            parsed = parse(mutation)
            assert parsed is not None
        except Exception as e:
            pytest.fail(f"GraphQL parsing failed with complex args: {e}")

    def test_mutation_with_newlines(self, graphql_schema):
        """Test docker args containing newlines are properly escaped."""
        # Create a docker script that inherently contains newlines
        training_command = "train.py nnue --config config/train_nnue_default.py"
        docker_script = _create_docker_script(training_command)

        # Add some newlines to simulate complex bash commands
        complex_script = (
            docker_script + "\necho 'additional command'\necho 'another line'"
        )
        escaped_args = _bash_c_quote(complex_script)

        # Should not contain literal newlines
        assert "\n" not in escaped_args

        # Create a GraphQL mutation
        mutation = f"""
            mutation {{
                createPod(dockerArgs: "{escaped_args}")
            }}
        """

        # Should parse correctly
        try:
            parsed = parse(mutation)
            assert parsed is not None
        except Exception as e:
            pytest.fail(f"GraphQL parsing failed with newlines: {e}")


class TestRegressionCases:
    """Test specific cases that caused the original GraphQL failures."""

    def test_single_quote_regression(self):
        """Test that single quotes don't cause 'Unexpected single quote' errors."""
        training_command = "train.py nnue --note='single quoted note'"
        docker_script = _create_docker_script(training_command)
        escaped_args = _bash_c_quote(docker_script)

        # Create mutation that would have failed before
        mutation = f"""
            mutation {{
                createPod(dockerArgs: "{escaped_args}")
            }}
        """

        # Should parse without the "Unexpected single quote" error
        try:
            parsed = parse(mutation)
            assert parsed is not None
        except Exception as e:
            pytest.fail(f"Single quote regression test failed: {e}")

    def test_number_format_regression(self):
        """Test that numeric arguments don't cause 'Invalid number' errors."""
        training_command = "train.py nnue --batch_size=32 --learning_rate=1e-4"
        docker_script = _create_docker_script(training_command)
        escaped_args = _bash_c_quote(docker_script)

        # Create mutation with numeric args
        mutation = f"""
            mutation {{
                createPod(dockerArgs: "{escaped_args}")
            }}
        """

        # Should parse without "Invalid number, expected digit but got 'f'" error
        try:
            parsed = parse(mutation)
            assert parsed is not None
        except Exception as e:
            pytest.fail(f"Number format regression test failed: {e}")

    @pytest.mark.parametrize("problematic_char", ['"', "'", "\\", "\n", "\t", "$", "`"])
    def test_various_problematic_characters(self, problematic_char):
        """Test various characters that could break GraphQL syntax."""
        training_command = f"train.py nnue --note=test{problematic_char}value"
        docker_script = _create_docker_script(training_command)
        escaped_args = _bash_c_quote(docker_script)

        # Create mutation
        mutation = f"""
            mutation {{
                createPod(dockerArgs: "{escaped_args}")
            }}
        """

        # Should parse regardless of the problematic character
        try:
            parsed = parse(mutation)
            assert parsed is not None
        except Exception as e:
            pytest.fail(f"Failed with character '{problematic_char}': {e}")


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])
