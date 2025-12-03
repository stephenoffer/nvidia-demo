"""Tests for Pythonic DataFrame API features.

Tests standard Python built-ins, operator overloading, indexing, and slicing.
"""

import pytest
import ray
from pipeline.api import PipelineDataFrame


@pytest.fixture(scope="module")
def ray_init():
    """Initialize Ray for tests."""
    ray.init(ignore_reinit_error=True)
    yield
    ray.shutdown()


@pytest.fixture
def sample_dataframe(ray_init):
    """Create a sample DataFrame for testing."""
    test_data = [
        {"id": i, "value": i * 2, "name": f"item_{i}"}
        for i in range(100)
    ]
    return PipelineDataFrame.from_dataset(ray.data.from_items(test_data))


class TestPythonBuiltins:
    """Test standard Python built-in support."""
    
    def test_len(self, sample_dataframe):
        """Test len() built-in."""
        assert len(sample_dataframe) == 100
    
    def test_iter(self, sample_dataframe):
        """Test iteration."""
        count = 0
        for row in sample_dataframe:
            assert isinstance(row, dict)
            assert "id" in row
            count += 1
            if count >= 10:
                break
        assert count == 10
    
    def test_bool(self, sample_dataframe):
        """Test bool() built-in."""
        assert bool(sample_dataframe) is True
        
        # Empty DataFrame should be falsy
        empty_df = PipelineDataFrame.from_dataset(ray.data.from_items([]))
        assert bool(empty_df) is False
    
    def test_contains(self, sample_dataframe):
        """Test 'in' operator."""
        # Check for row dict
        assert {"id": 0, "value": 0, "name": "item_0"} in sample_dataframe
        assert {"id": 999, "value": 999} not in sample_dataframe
        
        # Check for scalar value in any column
        assert 0 in sample_dataframe  # Should be in 'id' or 'value'
        assert "item_0" in sample_dataframe  # Should be in 'name'


class TestOperatorOverloading:
    """Test operator overloading."""
    
    def test_add_operator(self, sample_dataframe):
        """Test + operator for concatenation."""
        df1 = sample_dataframe[0:50]
        df2 = sample_dataframe[50:100]
        combined = df1 + df2
        
        assert len(combined) == 100
        assert combined.shape[0] == 100
    
    def test_or_operator(self, sample_dataframe):
        """Test | operator for union."""
        df1 = sample_dataframe[0:50]
        df2 = sample_dataframe[50:100]
        union = df1 | df2
        
        assert len(union) == 100
    
    def test_eq_operator(self, sample_dataframe):
        """Test == operator."""
        assert sample_dataframe == sample_dataframe  # Same object
        
        df_copy = sample_dataframe.copy()
        assert sample_dataframe != df_copy  # Different objects
    
    def test_ne_operator(self, sample_dataframe):
        """Test != operator."""
        df_copy = sample_dataframe.copy()
        assert sample_dataframe != df_copy


class TestIndexing:
    """Test Pythonic indexing and slicing."""
    
    def test_column_access(self, sample_dataframe):
        """Test column access with []."""
        column = sample_dataframe["id"]
        assert isinstance(column, list)
        assert len(column) == 100
        assert column[0] == 0
    
    def test_row_indexing(self, sample_dataframe):
        """Test row indexing with []."""
        row = sample_dataframe[0]
        assert isinstance(row, dict)
        assert row["id"] == 0
        assert row["value"] == 0
    
    def test_slicing(self, sample_dataframe):
        """Test row slicing."""
        sliced = sample_dataframe[0:10]
        assert len(sliced) == 10
        assert sliced.shape[0] == 10
        
        # Test with start only
        sliced_from = sample_dataframe[50:]
        assert len(sliced_from) == 50
    
    def test_multiple_column_selection(self, sample_dataframe):
        """Test multiple column selection."""
        selected = sample_dataframe[["id", "value"]]
        assert selected.columns == ["id", "value"]
        assert len(selected) == 100
    
    def test_attribute_access(self, sample_dataframe):
        """Test attribute-style column access."""
        column = sample_dataframe.id
        assert isinstance(column, list)
        assert len(column) == 100


class TestProperties:
    """Test Pandas-like properties."""
    
    def test_shape(self, sample_dataframe):
        """Test shape property."""
        shape = sample_dataframe.shape
        assert isinstance(shape, tuple)
        assert len(shape) == 2
        assert shape[0] == 100  # rows
        assert shape[1] == 3  # columns
    
    def test_columns(self, sample_dataframe):
        """Test columns property."""
        columns = sample_dataframe.columns
        assert isinstance(columns, list)
        assert "id" in columns
        assert "value" in columns
        assert "name" in columns
    
    def test_empty(self, sample_dataframe):
        """Test empty property."""
        assert sample_dataframe.empty is False
        
        empty_df = PipelineDataFrame.from_dataset(ray.data.from_items([]))
        assert empty_df.empty is True
    
    def test_copy(self, sample_dataframe):
        """Test copy() method."""
        df_copy = sample_dataframe.copy()
        assert len(df_copy) == len(sample_dataframe)
        assert df_copy.shape == sample_dataframe.shape
        assert df_copy.columns == sample_dataframe.columns
        assert df_copy != sample_dataframe  # Different objects


class TestIntegration:
    """Integration tests combining multiple Pythonic features."""
    
    def test_chain_operations(self, sample_dataframe):
        """Test chaining Pythonic operations."""
        result = (
            sample_dataframe
            .filter(lambda x: x["value"] > 50)
            [0:10]  # Slice after filter
        )
        assert len(result) <= 10
        assert result.shape[0] <= 10
    
    def test_operator_chaining(self, sample_dataframe):
        """Test chaining operators."""
        df1 = sample_dataframe[0:25]
        df2 = sample_dataframe[25:50]
        df3 = sample_dataframe[50:75]
        
        combined = df1 + df2 + df3
        assert len(combined) == 75
    
    def test_properties_after_operations(self, sample_dataframe):
        """Test properties work after transformations."""
        filtered = sample_dataframe.filter(lambda x: x["value"] > 50)
        
        assert isinstance(filtered.shape, tuple)
        assert isinstance(filtered.columns, list)
        assert filtered.empty == (len(filtered) == 0)

