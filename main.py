import IPython
from graph import save_network_html
from information import get_news_links
from processing import from_urls_to_kb


news_links = get_news_links("italian tech week", pages=3)
print(f"in total there are {len(news_links)} links")
kb = from_urls_to_kb(news_links)

filename = "kb.html"
save_network_html(kb, filename=filename)
# IPython.display.HTML(filename=filename)
