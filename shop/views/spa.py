from django.http import HttpResponse, HttpResponseNotAllowed


def spa_entry_view(request):
    if request.method not in ('GET', 'HEAD'):
        return HttpResponseNotAllowed(['GET', 'HEAD'])

    return HttpResponse(
        "TanlaAI backend API is running. Use /api/v1/ for API endpoints.",
        content_type='text/plain; charset=utf-8',
    )
