from bot import richmenu

WEB = "https://heat-map-frontend.vercel.app/map"


def test_build_richmenu_object_has_three_areas():
    obj = richmenu.build_richmenu_object(WEB)
    assert obj["size"] == {"width": 2500, "height": 843}
    assert len(obj["areas"]) == 3
    actions = [a["action"] for a in obj["areas"]]
    types = {a["type"] for a in actions}
    assert "uri" in types and "postback" in types


def test_build_richmenu_uri_points_to_website():
    obj = richmenu.build_richmenu_object(WEB)
    uri_actions = [a["action"] for a in obj["areas"] if a["action"]["type"] == "uri"]
    assert uri_actions[0]["uri"] == WEB


def test_build_richmenu_postback_actions():
    obj = richmenu.build_richmenu_object(WEB)
    data = {a["action"].get("data") for a in obj["areas"] if a["action"]["type"] == "postback"}
    assert data == {"action=weekly", "action=about"}
