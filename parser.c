#include <stdbool.h>
#include <stdio.h>

#include "tokenizer.h"
#include <base/arena.h>
#include <base/io.h>
#include "mlir_parser.h"

/*

// MLIR Structures
enum class AttributeKind { ATTR_INT, ATTR_STRING, ATTR_TYPE, ATTR_ARRAY, ATTR_DICT };
using AttrValue = std::variant<long long, std::string, struct Type*, std::vector<struct Attribute*>, std::map<std::string, struct Attribute*>>;
struct Attribute {
    AttributeKind kind;
    AttrValue value;
};

struct Type {
    std::string kind;
    std::map<std::string, Attribute*> attributes;
};

enum class ValueKind { BLOCK_ARG, OP_RESULT };
struct ValueRef {
    ValueKind kind;
    void* def; // Block* or Operation*
    size_t index;
};

struct Operation {
    std::string opname;
    std::vector<ValueRef> operands;
    std::vector<Type*> result_types;
    std::map<std::string, Attribute*> attributes;
    std::vector<struct Region*> regions;
    std::vector<struct Block*> successors;
};

struct Block {
    std::vector<Type*> arg_types;
    std::vector<Operation*> operations;
};

struct Region {
    std::vector<Block*> blocks;
};

*/

// Token
/*
enum class TokenType {
    TOKEN_EOF,
    TOKEN_IDENTIFIER,
    TOKEN_VALUE_ID,
    TOKEN_BLOCK_LABEL,
    TOKEN_INTEGER,
    TOKEN_STRING,
    TOKEN_PUNCTUATION,
    TOKEN_ARROW
};
*/

/*
struct Token {
    TokenType type;
    std::string str_value;
    long long int_value = 0;
};

std::string tokentype_to_string(TokenType tt) {
#define CASE_TOKEN(x) case TokenType::x: return #x;
    switch (tt) {
        CASE_TOKEN(TOKEN_EOF)
        CASE_TOKEN(TOKEN_IDENTIFIER)
        CASE_TOKEN(TOKEN_VALUE_ID)
        CASE_TOKEN(TOKEN_BLOCK_LABEL)
        CASE_TOKEN(TOKEN_INTEGER)
        CASE_TOKEN(TOKEN_STRING)
        CASE_TOKEN(TOKEN_PUNCTUATION)
        CASE_TOKEN(TOKEN_ARROW)
        default: abort();
    }
}


// Lexer
class Lexer {
    const std::string &input;
    size_t pos = 0; // Points to the next (untokenized) character
public:
    Lexer(const std::string& input) : input(input) {}
    Token get_next_token() {
        while (pos < input.size() && std::isspace(input[pos])) pos++;
        if (pos >= input.size()) return {TokenType::TOKEN_EOF, ""};

        char c = input[pos];
        if (std::isalpha(c) || c == '_') {
            size_t start = pos++;
            while (pos < input.size() && (std::isalnum(input[pos]) || input[pos] == '_' || input[pos] == '.')) pos++;
            return {TokenType::TOKEN_IDENTIFIER, input.substr(start, pos - start)};
        }
        if (c == '%') {
            pos++;
            size_t start = pos;
            while (pos < input.size() && std::isalnum(input[pos])) pos++;
            return {TokenType::TOKEN_VALUE_ID, input.substr(start, pos - start)};
        }
        if (c == '^') {
            pos++;
            size_t start = pos;
            while (pos < input.size() && std::isalnum(input[pos])) pos++;
            return {TokenType::TOKEN_BLOCK_LABEL, input.substr(start, pos - start)};
        }
        if (std::isdigit(c)) {
            size_t start = pos;
            while (pos < input.size() && std::isdigit(input[pos])) pos++;
            return {TokenType::TOKEN_INTEGER, "", std::stoll(input.substr(start, pos - start))};
        }
        if (c == '"') {
            pos++;
            size_t start = pos;
            while (pos < input.size() && input[pos] != '"') pos++;
            if (pos >= input.size() || input[pos] != '"') throw std::runtime_error("Unterminated string");
            std::string value = input.substr(start, pos - start);
            pos++;
            return {TokenType::TOKEN_STRING, value};
        }
        if (std::string("{}()[],:=<>").find(c) != std::string::npos) {
            pos++;
            return {TokenType::TOKEN_PUNCTUATION, std::string(1, c)};
        }
        if (c == '-') {
            pos++;
            if (pos >= input.size()) throw std::runtime_error("Unfinished ->");
            c = input[pos];
            if (c == '>') {
                pos++;
                return {TokenType::TOKEN_ARROW, "->"};
            } else {
                throw std::runtime_error("> expected");
            }
        }
        throw std::runtime_error("Unknown character");
    }
};

// Symbol Table
class SymbolTable {
    std::vector<std::unordered_map<std::string, ValueRef>> scopes;
public:
    void push_scope() { scopes.emplace_back(); }
    void pop_scope() { if (!scopes.empty()) scopes.pop_back(); }
    void add_symbol(const std::string& key, ValueRef value) {
        if (scopes.empty()) throw std::runtime_error("No scope");
        scopes.back()[key] = value;
    }
    ValueRef* lookup_symbol(const std::string& key) {
        for (auto it = scopes.rbegin(); it != scopes.rend(); ++it) {
            auto found = it->find(key);
            if (found != it->end()) return &found->second;
        }
        throw std::runtime_error("Undefined symbol: " + key);
    }
};

// Parser
class Parser {
    Lexer lexer;
    Token current_token;
    SymbolTable symbol_table;

    void consume_token() { current_token = lexer.get_next_token(); }
    void expect_token(TokenType type, const std::string& value = "") {
        if (current_token.type != type || (!value.empty() && current_token.value != value)) {
            throw std::runtime_error("Expected " + (value.empty() ? "token" : value));
        }
        consume_token();
    }

public:
    Parser(const std::string& input) : lexer(input) { current_token = lexer.get_next_token(); }

    Attribute* parse_attribute() {
        if (current_token.type == TokenType::TOKEN_INTEGER) {
            auto attr = new Attribute{AttributeKind::ATTR_INT, current_token.int_value};
            consume_token();
            return attr;
        }
        if (current_token.type == TokenType::TOKEN_STRING) {
            auto attr = new Attribute{AttributeKind::ATTR_STRING, current_token.value};
            consume_token();
            return attr;
        }
        if (current_token.type == TokenType::TOKEN_IDENTIFIER) {
            auto type = parse_type();
            return new Attribute{AttributeKind::ATTR_TYPE, type};
        }
        if (current_token.type == TokenType::TOKEN_PUNCTUATION && current_token.value == "[") {
            consume_token();
            std::vector<Attribute*> elements;
            while (current_token.type != TokenType::TOKEN_PUNCTUATION || current_token.value != "]") {
                elements.push_back(parse_attribute());
                if (current_token.value == ",") consume_token();
            }
            expect_token(TokenType::TOKEN_PUNCTUATION, "]");
            return new Attribute{AttributeKind::ATTR_ARRAY, elements};
        }
        if (current_token.type == TokenType::TOKEN_PUNCTUATION && current_token.value == "{") {
            consume_token();
            std::map<std::string, Attribute*> dict;
            while (current_token.type != TokenType::TOKEN_PUNCTUATION || current_token.value != "}") {
                expect_token(TokenType::TOKEN_IDENTIFIER);
                std::string key = current_token.value;
                consume_token();
                expect_token(TokenType::TOKEN_PUNCTUATION, "=");
                dict[key] = parse_attribute();
                if (current_token.value == ",") consume_token();
            }
            expect_token(TokenType::TOKEN_PUNCTUATION, "}");
            return new Attribute{AttributeKind::ATTR_DICT, dict};
        }
        throw std::runtime_error("Invalid attribute");
    }

    Type* parse_type() {
        expect_token(TokenType::TOKEN_IDENTIFIER);
        auto type = new Type{current_token.value, {}};
        consume_token();
        if (current_token.value == "<") {
            consume_token();
            while (current_token.value != ">") {
                expect_token(TokenType::TOKEN_IDENTIFIER);
                std::string key = current_token.value;
                consume_token();
                expect_token(TokenType::TOKEN_PUNCTUATION, "=");
                type->attributes[key] = parse_attribute();
                if (current_token.value == ",") consume_token();
            }
            expect_token(TokenType::TOKEN_PUNCTUATION, ">");
        }
        return type;
    }

    Operation* parse_operation() {
        auto op = new Operation{{}, {}, {}, {}, {}, {}};
        if (current_token.type == TokenType::TOKEN_VALUE_ID) {
            do {
                std::string name = current_token.value;
                consume_token();
                ValueRef ref{ValueKind::OP_RESULT, op, op->result_types.size()};
                symbol_table.add_symbol(name, ref);
                op->result_types.push_back(nullptr); // Filled later
                if (current_token.value == ",") consume_token();
            } while (current_token.type == TokenType::TOKEN_VALUE_ID);
            expect_token(TokenType::TOKEN_PUNCTUATION, "=");
        }
        expect_token(TokenType::TOKEN_STRING);
        op->opname = current_token.value;
        //consume_token();
        expect_token(TokenType::TOKEN_PUNCTUATION, "(");
        while (current_token.value != ")") {
            expect_token(TokenType::TOKEN_VALUE_ID);
            op->operands.push_back(*symbol_table.lookup_symbol(current_token.value));
            consume_token();
            if (current_token.value == ",") consume_token();
        }
        expect_token(TokenType::TOKEN_PUNCTUATION, ")");
        if (current_token.value == "{") {
            consume_token();
            while (current_token.value != "}") {
                expect_token(TokenType::TOKEN_IDENTIFIER);
                std::string key = current_token.value;
                //consume_token();
                expect_token(TokenType::TOKEN_PUNCTUATION, "=");
                op->attributes[key] = parse_attribute();
                if (current_token.value == ",") consume_token();
            }
            expect_token(TokenType::TOKEN_PUNCTUATION, "}");
        }
        expect_token(TokenType::TOKEN_PUNCTUATION, ":");
        expect_token(TokenType::TOKEN_PUNCTUATION, "(");
        while (current_token.value != ")") {
            op->result_types.push_back(parse_type());
            if (current_token.value == ",") consume_token();
        }
        expect_token(TokenType::TOKEN_PUNCTUATION, ")");
        expect_token(TokenType::TOKEN_PUNCTUATION, "->");
        op->result_types[0] = parse_type();
        return op;
    }

    Block* parse_block() {
        auto block = new Block{{}, {}};
        if (current_token.type == TokenType::TOKEN_BLOCK_LABEL) {
            consume_token();
            expect_token(TokenType::TOKEN_PUNCTUATION, ":");
        }
        while (current_token.type != TokenType::TOKEN_EOF && current_token.value != "}") {
            block->operations.push_back(parse_operation());
        }
        return block;
    }

    Region* parse_region() {
        expect_token(TokenType::TOKEN_PUNCTUATION, "{");
        symbol_table.push_scope();
        std::vector<Block*> blocks;
        while (current_token.value != "}") {
            blocks.push_back(parse_block());
        }
        expect_token(TokenType::TOKEN_PUNCTUATION, "}");
        symbol_table.pop_scope();
        return new Region{blocks};
    }

    Operation* parse_module() {
        expect_token(TokenType::TOKEN_IDENTIFIER, "module");
        expect_token(TokenType::TOKEN_PUNCTUATION, "{");
        symbol_table.push_scope();
        std::vector<Operation*> ops;
        while (current_token.value != "}") {
            ops.push_back(parse_operation());
        }
        expect_token(TokenType::TOKEN_PUNCTUATION, "}");
        symbol_table.pop_scope();
        auto block = new Block{{}, ops};
        auto region = new Region{{block}};
        return new Operation{"builtin.module", {}, {}, {}, {region}, {}};
    }
};
*/

void tokenizer_print_all_tokens(Arena *arena, const string input_code) {
    unsigned char *string_start;
    string_start = (unsigned char*)input_code.str;
    uint64_t cur=0;
    while (true) {
        TokenType token_type;
        uint64_t first, last;
        first = cur;
        tokenizer_get_next_token(string_start, &cur, &token_type);
        last = cur-1;
        bool debug = false;
        if (debug) {
            string token_str = str_substr(input_code, first, last-first+1);
            if (token_type == TK_WHITESPACE || token_type == TK_NEWLINE
                    || token_type == TK_EOF) {
                token_str = str_lit("");
            }
            println(arena, str_lit("Token({}, \"{}\", {},{})"),
                tokentype_to_string(token_type), token_str, first, last);
        }
        if (token_type == TK_EOF) {
            return;
        }
    }
}

string print_operation(Arena *arena, int indent_level, Operation *op);
static const char* op_type_to_string(OpType type);

string indent(Arena *arena, int indent_level) {
    const int indent_spaces=4;
    int buf_size=indent_level*indent_spaces;
    char* buf = arena_alloc_array(arena, char, buf_size);
    for (int64_t i = 0; i < buf_size; i++) {
        buf[i] = ' ';
    }
    string str = {buf, buf_size};
    return str;
}

string print_block(Arena *arena, int bb_index, int indent_level, Block *block) {
    string result = format(arena, str_lit("{}^bb{}\n"),
            indent(arena, indent_level), bb_index);
    for (int i=0; i < block->n_operations; i++) {
        result = str_concat(arena, result,
            print_operation(arena, indent_level+1, block->operations[i])
            );
    }
    return result;
}

string print_region(Arena *arena, int indent_level, Region *region) {
    string result = str_lit("");
    result = str_concat(arena, result, str_lit("{\n"));
    for (int i=0; i < region->n_blocks; i++) {
        result = str_concat(arena, result,
            print_block(arena, i, indent_level, region->blocks[i])
            );
    }
    result = str_concat(arena, result, indent(arena, indent_level));
    result = str_concat(arena, result, str_lit("}"));
    return result;
}

int reg_idx=0;

static const char* op_type_to_string(OpType type) {
    switch (type) {
        case OP_TYPE_UNREGISTERED: return "unregistered";
        case OP_TYPE_MODULE: return "module";
        case OP_TYPE_ARITH_ADDI: return "arith.addi";
        case OP_TYPE_ARITH_SUBI: return "arith.subi";
        case OP_TYPE_ARITH_MULI: return "arith.muli";
        case OP_TYPE_ARITH_DIVI: return "arith.divi";
        case OP_TYPE_ARITH_ADDF: return "arith.addf";
        case OP_TYPE_ARITH_SUBF: return "arith.subf";
        case OP_TYPE_ARITH_MULF: return "arith.mulf";
        case OP_TYPE_ARITH_DIVF: return "arith.divf";
        case OP_TYPE_ARITH_CONSTANT: return "arith.constant";
        case OP_TYPE_ARITH_CMPI: return "arith.cmpi";
        case OP_TYPE_ARITH_CMPF: return "arith.cmpf";
        case OP_TYPE_MEMREF_LOAD: return "memref.load";
        case OP_TYPE_MEMREF_STORE: return "memref.store";
        case OP_TYPE_MEMREF_ALLOC: return "memref.alloc";
        case OP_TYPE_MEMREF_DEALLOC: return "memref.dealloc";
        case OP_TYPE_CF_BR: return "cf.br";
        case OP_TYPE_CF_COND_BR: return "cf.cond_br";
        case OP_TYPE_CF_SWITCH: return "cf.switch";
        case OP_TYPE_FUNC_FUNC: return "func.func";
        case OP_TYPE_FUNC_RETURN: return "func.return";
        case OP_TYPE_FUNC_CALL: return "func.call";
        case OP_TYPE_SCF_FOR: return "scf.for";
        case OP_TYPE_SCF_WHILE: return "scf.while";
        case OP_TYPE_SCF_IF: return "scf.if";
        default: return "unknown";
    }
}

string print_operation(Arena *arena, int indent_level, Operation *op) {
    string result = indent(arena, indent_level);
    
    // Print results if any
    if (op->n_result_types > 0) {
        for (int i = 0; i < op->n_result_types; i++) {
            if (i > 0) result = str_concat(arena, result, str_lit(", "));
            result = str_concat(arena, result, format(arena, str_lit("%{}"), reg_idx + i));
        }
        result = str_concat(arena, result, str_lit(" = "));
        reg_idx += op->n_result_types;
    }
    
    // Print operation name
    if (op->op_type == OP_TYPE_UNREGISTERED) {
        result = str_concat(arena, result, format(arena, str_lit("\"{}\""), op->opname));
    } else {
        result = str_concat(arena, result, str_from_cstr_view((char*)op_type_to_string(op->op_type)));
    }
    
    // Print operands with types (always include parentheses)
    result = str_concat(arena, result, str_lit("("));
    for (int i = 0; i < op->n_operands; i++) {
        if (i > 0) result = str_concat(arena, result, str_lit(", "));
        ValueRef *operand = op->operands[i];
        if (operand && operand->type) {
            // For now, use a simple format for operands
            result = str_concat(arena, result, format(arena, str_lit("%{}: {}"), 
                                                    (int64_t)operand->result_index, operand->type->str));
        } else {
            result = str_concat(arena, result, str_lit("%?"));
        }
    }
    result = str_concat(arena, result, str_lit(")"));
    
    // Print attributes if any
    if (op->n_attributes > 0) {
        result = str_concat(arena, result, str_lit(" {"));
        for (int i = 0; i < op->n_attributes; i++) {
            if (i > 0) result = str_concat(arena, result, str_lit(", "));
            Attribute *attr = op->attributes[i];
            result = str_concat(arena, result, format(arena, str_lit("{} = "), str_from_cstr_view((char*)attr->name)));
            switch (attr->kind) {
                case ATTR_KIND_INTEGER:
                    result = str_concat(arena, result, format(arena, str_lit("{}"), attr->data.integer_value));
                    break;
                case ATTR_KIND_STRING:
                    result = str_concat(arena, result, format(arena, str_lit("\"{}\""), str_from_cstr_view((char*)attr->data.string_value)));
                    break;
                default:
                    result = str_concat(arena, result, str_lit("..."));
            }
        }
        result = str_concat(arena, result, str_lit("}"));
    }
    
    // Print result types if any
    if (op->n_result_types > 0) {
        result = str_concat(arena, result, str_lit(" -> "));
        for (int i = 0; i < op->n_result_types; i++) {
            if (i > 0) result = str_concat(arena, result, str_lit(", "));
            if (op->result_types && op->result_types[i]) {
                result = str_concat(arena, result, op->result_types[i]->str);
            } else {
                result = str_concat(arena, result, str_lit("?"));
            }
        }
    }
    
    // Print regions if any
    if (op->n_regions > 0) {
        result = str_concat(arena, result, str_lit(" "));
        for (int i = 0; i < op->n_regions; i++) {
            result = str_concat(arena, result,
                print_region(arena, indent_level, op->regions[i])
                );
        }
    }
    
    result = str_concat(arena, result, str_lit("\n"));
    return result;
}

// Main
int main(int argc, char *argv[]) {
    string mlir_code = str_lit("module {\n"
                            "  %0 = \"std.constant\"() {value = 42} : () -> i32\n"
                            "  \"std.return\"(%0) : (i32) -> ()\n"
                            "}");
    Arena *arena = arena_create(10*1024*1024);
    if (argc == 2) {
        mlir_code = read_file_ok(arena, str_from_cstr_view(argv[1]));
    }
    tokenizer_print_all_tokens(arena, mlir_code);

    Parser parser;
    parser_init(arena, &parser, mlir_code);
    // Uncomment to run parser (will currently fail with a syntax error):
    Operation* op = parse_module(&parser);
    println(arena, str_lit("MLIR:"));
    println(arena, str_lit("{}"), print_operation(arena, 0, op));

    arena_free(arena);
    return 0;
}
