use std::collections::HashMap;

/// A template for AI prompts that supports variable substitution.
pub struct PromptTemplate {
    template: String,
}

impl PromptTemplate {
    pub const fn new(template: String) -> Self {
        Self { template }
    }

    /// Render the template by replacing `{{key}}` with the corresponding value.
    pub fn render(&self, vars: &HashMap<&str, &str>) -> String {
        let mut output = self.template.to_string();
        for (k, v) in vars {
            let placeholder = format!("{{{{{}}}}}", k);
            output = output.replace(&placeholder, v);
        }
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_render() {
        let t = PromptTemplate::new("Hello {{name}}, welcome to {{place}}!".to_string());
        let mut vars = HashMap::new();
        vars.insert("name", "Alice");
        vars.insert("place", "Wonderland");
        assert_eq!(t.render(&vars), "Hello Alice, welcome to Wonderland!");
    }
}
