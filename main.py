from src.services.recommendation_service import RecommendationService

def main():
    service = RecommendationService(sample_size=10000)
    service.initialize()
    
    print("\ Type a movie, genre, or description (or 'quit'):")
    
    while True:
        query = input("\n> ").strip()
        if query.lower() == 'quit':
            break
        
        try:
            recs = service.get_recommendations(title=query, top_k=5)
        except:
            try:
                recs = service.get_recommendations(description=query, top_k=5)
            except:
                try:
                    recs = service.get_recommendations(genre=query, top_k=5)
                except:
                    print("No results found")
                    continue
        
        print(f"\nTop 5 recommendations for '{query}':")
        for i, rec in enumerate(recs, 1):
            print(f"{i}. {rec['title']} (Score: {rec['similarity_score']:.3f})")

if __name__ == "__main__":
    main()