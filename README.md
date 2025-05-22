# MLIR

Faster and simpler MLIR infrastructure.

# Tokenizer Design

* Just return token type, first, last. Probably first/last should be integer
  indices into the string, not pointers.
* The tokenizer is thus as efficient as it can be, since we have to return
  location information and token type anyway and we do not return nor do
  anything else.
* The API surface to use the tokenizer is also minimal, and the tokenizer code
  can be easily reused / adapted for any other tokenizer easily, since just the
  list of tokens enum and the re2c code needs to be updated.
* We try to return as much information as we can as part of the token type,
  such as `TK_PLUS`, `TK_MINUS`, `TK_LEFT_PAREN`, etc., that way we do not need
  to consult the string at all for those. Only for tokens like string or number
  we need to actually handle those in some way in the parser, although even
  those we can postpone converting them until later in the semantic phase.
* One can skip whitespace (both first/last needed), but can also return it as a
  token (just `last` needed, but not a huge overhead to return `first` also,
  making the parser independent if we skip characters or not in the tokenizer).
* In the parser's actions we can extract the string using first/last and
  convert / further process as needed based on the syntax information from the
  parser or possibly even convert later based on semantic information.
