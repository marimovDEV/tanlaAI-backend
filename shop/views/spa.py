from pathlib import Path

from django.conf import settings
from django.http import HttpResponse, HttpResponseNotAllowed


def spa_entry_view(request):
    if request.method not in ('GET', 'HEAD'):
        return HttpResponseNotAllowed(['GET', 'HEAD'])

    index_path = Path(settings.BASE_DIR) / 'static' / 'react' / 'index.html'
    if not index_path.exists():
        return HttpResponse(
            "React build topilmadi. `frontend-react` ichida `npm run build` ni ishga tushiring.",
            status=503,
            content_type='text/plain; charset=utf-8',
        )

    return HttpResponse(index_path.read_text(encoding='utf-8'), content_type='text/html; charset=utf-8')
