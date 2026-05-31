from graf.models.gcn_risk import build_model


def test_build_model_has_expected_repr_or_modules():
    model = build_model(in_channels=12, hidden_channels=16)
    if hasattr(model, "conv1"):
        assert model.head.out_features == 1
    else:
        text = repr(model)
        assert "SimpleGCNRiskModel" in text
        assert "in_channels=12" in text
        assert "hidden_channels=16" in text
