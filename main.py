import functions_framework

import swiss
import sheet_manager


@functions_framework.http
def generate_pairings(request):
    req = request.get_json(silent=True)
    if not req:
        return "Invalid Request", 400

    for f in ["sheet", "cycle"]:
        if f not in req:
            return f'Request is missing "{f}"', 400

    sheet = sheet_manager.UrlSheetManager(req["sheet"], req["cycle"], True)
    sheet_old = None
    if req.get("sheet_old"):
        sheet_old = sheet_manager.UrlSheetManager(req["sheet_old"], 5, True)
    cycle = int(req["cycle"])
    write = req.get("write", False) == True
    swiss.GeneratePairings(sheet, sheet_old, cycle, tabprint=False, write=write)
    return "OK"
