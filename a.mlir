module {
  %0 = \"std.constant\"() {value = 42} : () -> i32
  \"std.return\"(%0) : (i32) -> ()
}
