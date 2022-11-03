import wikipedia
from GoogleNews import GoogleNews
from newspaper import Article


def get_wikipedia_data(candidate_entity):
    try:
        page = wikipedia.page(candidate_entity, auto_suggest=False)
        entity_data = {
            "title": page.title,
            "url": page.url,
            "summary": page.summary
        }
        return entity_data
    except:
        return None


def get_article(url):
    article = Article(url)
    article.download()
    article.parse()
    return article


def get_news_links(query, lang="en", region="US", pages=1, max_links=100000):
    news = GoogleNews(lang=lang, region=region)
    news.search(query)

    # set to not duplicate any link
    all_urls = []
    for page in range(pages):
        news.get_page(page)
        all_urls += news.get_links()

    # return only up a maximum number of links after eliminating the many duplicates
    return list(set(all_urls))[:max_links]
