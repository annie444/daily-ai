#[cfg(test)]
mod tests {
    use crate::ai::ResponseCleaner;

    #[test]
    fn cleans_nested_json() {
        let resp = "{\"outer\": {\"inner\": 1}, \"next\": 2}";
        let mut cleaner = ResponseCleaner::new(resp);
        let cleaned = cleaner.clean();
        assert_eq!(cleaned, "{\"outer\":{\"inner\":1},\"next\":2}");
    }

    #[test]
    fn cleans_negative_numbers() {
        let resp = "{\"temp\": -5.0}";
        let mut cleaner = ResponseCleaner::new(resp);
        let cleaned = cleaner.clean();
        assert_eq!(cleaned, "{\"temp\":-5.0}");
    }

    #[test]
    fn cleans_booleans_and_null() {
        let resp = "{\"is_valid\": true, \"is_bad\": false, \"nothing\": null}";
        let mut cleaner = ResponseCleaner::new(resp);
        let cleaned = cleaner.clean();
        assert_eq!(
            cleaned,
            "{\"is_valid\":true,\"is_bad\":false,\"nothing\":null}"
        );
    }

    #[test]
    fn cleans_noise_around_values() {
        let resp = "!{\"key\": O\"Value\", \"num\": #123, \"bool\": !true}";
        let mut cleaner = ResponseCleaner::new(resp);
        let cleaned = cleaner.clean();
        assert_eq!(cleaned, "{\"key\":\"Value\",\"num\":123,\"bool\":true}");
    }
}
